from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "yolo11n.pt"
DATA_YAML = ROOT / "dataset" / "cedulas" / "data.yaml"
EPOCHS = 100
IMAGE_SIZE = 640
BATCH_SIZE = 8


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo base nao encontrado: {MODEL_PATH}")
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset nao encontrado: {DATA_YAML}. Rode script.py antes.")

    model = YOLO(str(MODEL_PATH))
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        project=str(ROOT / "runs"),
        name="cedulas_yolo11n",
        pretrained=True,
        device="cpu",
        workers=0,
    )


if __name__ == "__main__":
    main()
