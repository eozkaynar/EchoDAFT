import click

from echodaft.scripts.video_daft import run as daft

@click.group()
def cli():
    """EchoDAFT command line interface."""
    pass

cli.add_command(daft, name="daft")

if __name__ == "__main__":
    cli()