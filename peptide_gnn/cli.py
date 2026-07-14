import tensorflow as tf

import click
import shutil
import pathlib
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

from contextlib import contextmanager
from rich import console

from peptide_gnn.util import load_dataset
from peptide_gnn.util import shuffle_dataset
from peptide_gnn.util import split_dataset

from peptide_gnn.util import write_records
from peptide_gnn.util import read_records

from peptide_gnn.util import create_featurizer
from peptide_gnn.util import create_model
from peptide_gnn.util import create_optimizer

from peptide_gnn.util import compute_saliency

from peptide_gnn.conf import DEFAULT_INPUT_DIR
from peptide_gnn.conf import DEFAULT_OUTPUT_DIR


console = console.Console()


@click.group()
def cli() -> None:
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass
    _trigger_tf_logging()


@cli.command("run")
@click.argument("input-dir", default=DEFAULT_INPUT_DIR, type=str)
@click.option("--output-dir", "-o", default=DEFAULT_OUTPUT_DIR, type=str)
@click.option("--learning-rate", default=1e-4, type=float)
@click.option("--end-learning-rate", default=1e-6, type=float)
@click.option("--epochs", default=100, type=int)
@click.option("--batch-size",  default=16, type=int)
@click.option("--reshuffle-each-epoch", is_flag=True, default=False, type=bool)
@click.option("--save-model", is_flag=True, default=False, type=bool)
@click.option("--random-seed", default=42, type=int)
def run(
    input_dir, 
    output_dir, 
    learning_rate, 
    end_learning_rate, 
    epochs, 
    batch_size, 
    reshuffle_each_epoch, 
    save_model,
    random_seed,
) -> None:
    
    input_dir = pathlib.Path(input_dir)
    output_dir = pathlib.Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    console.print("\n[white bold]Run(s) starting\n")
    
    if input_dir.is_file():
        input_files = [input_dir]
        console.print(f"Input data: [magenta]{pathlib.Path(input_dir).absolute()}/[white]{input_dir.name}")
    else:
        input_files = input_dir.glob('*.csv')
        console.print(f"Input data: [magenta]{pathlib.Path(input_dir).absolute()}/[white]*.csv")
        
    console.print(f"Output data: [magenta]{pathlib.Path(output_dir).absolute()}/")

    for file in input_files:
        run_model(
            file, 
            output_dir=output_dir, 
            learning_rate=learning_rate, 
            end_learning_rate=end_learning_rate, 
            epochs=epochs, 
            batch_size=batch_size, 
            reshuffle_each_epoch=reshuffle_each_epoch,
            save_model=save_model, 
            random_seed=random_seed
        )

    console.print("\n[white bold]Run(s) done")

@contextmanager
def task_status(message, spinner="dots"):
    with console.status(f"[cyan]{message}", spinner=spinner):
        yield
        console.print(f"[green]✔[/green] [cyan]{message}")

def run_model(
    input_file, 
    output_dir, 
    learning_rate, 
    end_learning_rate, 
    epochs, 
    batch_size, 
    reshuffle_each_epoch, 
    save_model, 
    random_seed
):
    if (output_dir / f"{input_file.stem}_prediction.csv").exists():
        return 
    
    click.secho(f"\nRunning {input_file}", fg='green', nl=True)

    record_dir = output_dir / 'temp/' # TF records will be removed after the run
    
    with task_status("Preparing datasets"):
        x, y = load_dataset(input_file)
        x, y = shuffle_dataset(x, y, seed=random_seed)
        (x_train, x_val, x_test), (y_train, y_val, y_test) = split_dataset(x, y)

    with task_status("Processing datasets"):
        featurizer = create_featurizer()

        write_records(x_train, y_train, featurizer, record_dir / 'train')
        write_records(x_val, y_val, featurizer, record_dir / 'val')
        write_records(x_test, y_test, featurizer, record_dir / 'test')

        train_ds = read_records(
            record_dir / 'train', batch_size=batch_size, shuffle=reshuffle_each_epoch
        )
        val_ds = read_records(
            record_dir / 'val', batch_size=batch_size, shuffle=False
        )
        test_ds = read_records(
            record_dir / 'test', batch_size=batch_size, shuffle=False
        )

    with task_status("Creating model"):
        unshuffled_train_ds = read_records(record_dir / 'train', batch_size=batch_size, shuffle=False)
        ds = unshuffled_train_ds.concatenate(val_ds.concatenate(test_ds))
        model = create_model(ds)
        decay_steps = epochs * (len(x_train) // batch_size)
        optimizer = create_optimizer(learning_rate, end_learning_rate, decay_steps)
        loss = tf.keras.losses.Huber()
        model.compile(optimizer, loss, metrics=["mse", "mae"])

    with task_status("Training model"):
        model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=0)

    with task_status("Explaining model"):
        atom_maps = compute_saliency(model, ds.unbatch().batch(256))
        atom_maps = dict(zip(x, atom_maps))
        
    with task_status("Making predictions"):
        df_result = pd.DataFrame({
            'peptide': x,
            'observed_rt': y,
            'predicted_rt': model.predict(ds, verbose=0).squeeze(),
            'set_type': ["train"] * len(x_train) + ["val"] * len(x_val) + ["test"] * len(x_test)
        })
        if save_model:
            model_dir = output_dir / 'models/'
            model_dir.mkdir(parents=True, exist_ok=True)
            model.save(model_dir / f'{input_file.stem}_model.keras')

    with task_status(f"Saving results to {output_dir}"):
        df_result.to_csv(output_dir / f'{input_file.stem}_prediction.csv', index=False)
        with open(output_dir / f'{input_file.stem}_saliency.pkl', "wb") as f:
            pickle.dump(atom_maps, f)

    if record_dir.exists() and record_dir.is_dir():
        shutil.rmtree(record_dir)

    tf.keras.backend.clear_session()

    click.secho(f"Done", fg='green', nl=True)

def _trigger_tf_logging():
    tf.convert_to_tensor([42])
    model = tf.keras.Sequential([tf.keras.layers.Dense(1)])
    model.compile('sgd', 'mse')
    model.fit(tf.ones((1, 4)), tf.ones((1,)), verbose=0)


if __name__ == "__main__":
    cli()
