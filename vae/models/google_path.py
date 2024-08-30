import logging
import pathlib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def return_output_folder() -> pathlib.Path:
    try:
        from google.colab import drive

        log.info("Running on Google Colab.")
        save_path = "/content/drive/MyDrive/thesis/data/"
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
        return pathlib.Path(save_path)

    except ImportError:
        log.info("Not running on Google Colab.")
        path = "/Users/joachim/Library/Mobile Documents/com~apple~CloudDocs/thesis/data"
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        return pathlib.Path(path)
