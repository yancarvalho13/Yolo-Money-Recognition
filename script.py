from pathlib import Path
import random
import re
import shutil

SEED = 42
DATASET_ROOT = Path(r"C:\Users\yan.carvalho\Pictures\Cedulas DataSet")
EXPORT_ROOT = DATASET_ROOT / "project-1-at-2026-04-27-16-45-40854fb9"
OUTPUT_ROOT = Path(__file__).resolve().parent / "dataset" / "cedulas"
SPLIT_RATIO = 0.8


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def build_image_index():
    image_index = {}
    for image_path in DATASET_ROOT.iterdir():
        if not image_path.is_file() or image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        aliases = {normalize_name(image_path.stem)}
        if "-" in image_path.stem:
            aliases.add(normalize_name(image_path.stem.split("-", 1)[1]))

        for alias in aliases:
            image_index[alias] = image_path

    return image_index


def prepare_output_dirs():
    for subset in ("train", "val"):
        (OUTPUT_ROOT / "images" / subset).mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / "labels" / subset).mkdir(parents=True, exist_ok=True)


def write_data_yaml(classes: list[str]):
    yaml_path = OUTPUT_ROOT / "data.yaml"
    names_block = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(classes))
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {OUTPUT_ROOT.as_posix()}",
                "train: images/train",
                "val: images/val",
                "",
                "names:",
                names_block,
                "",
            ]
        ),
        encoding="utf-8",
    )


def main():
    random.seed(SEED)
    prepare_output_dirs()

    image_index = build_image_index()
    label_paths = sorted((EXPORT_ROOT / "labels").glob("*.txt"))
    classes = [line.strip() for line in (EXPORT_ROOT / "classes.txt").read_text(encoding="utf-8").splitlines() if line.strip()]

    matched_pairs = []
    missing_images = []

    for label_path in label_paths:
        label_key = normalize_name(label_path.stem.split("-", 1)[1] if "-" in label_path.stem else label_path.stem)
        image_path = image_index.get(label_key)
        if image_path is None:
            missing_images.append(label_path.name)
            continue
        matched_pairs.append((image_path, label_path))

    if not matched_pairs:
        raise RuntimeError("Nenhuma label foi associada a uma imagem. Verifique os nomes do export.")

    random.shuffle(matched_pairs)
    split_index = max(1, int(len(matched_pairs) * SPLIT_RATIO))
    train_pairs = matched_pairs[:split_index]
    val_pairs = matched_pairs[split_index:]

    if not val_pairs:
        val_pairs = train_pairs[-1:]
        train_pairs = train_pairs[:-1]

    for subset, pairs in (("train", train_pairs), ("val", val_pairs)):
        for image_path, label_path in pairs:
            shutil.copy2(image_path, OUTPUT_ROOT / "images" / subset / image_path.name)
            shutil.copy2(label_path, OUTPUT_ROOT / "labels" / subset / f"{image_path.stem}.txt")

    write_data_yaml(classes)

    print(f"Dataset YOLO preparado em: {OUTPUT_ROOT}")
    print(f"Treino: {len(train_pairs)} imagens")
    print(f"Validação: {len(val_pairs)} imagens")

    if missing_images:
        print("Labels sem imagem correspondente:")
        for item in missing_images:
            print(f"- {item}")


if __name__ == "__main__":
    main()
