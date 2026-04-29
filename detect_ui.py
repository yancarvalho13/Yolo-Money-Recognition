from pathlib import Path
import sys

import cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "runs" / "cedulas_yolo11n" / "weights" / "best.pt"
IMAGE_FILTER = "Imagens (*.png *.jpg *.jpeg *.bmp)"


class DetectionWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Deteccao de Cedulas")
        self.resize(1200, 900)

        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Modelo nao encontrado em {MODEL_PATH}")

        self.model = YOLO(str(MODEL_PATH))
        self.current_pixmap: QPixmap | None = None

        self.select_button = QPushButton("Selecionar imagem")
        self.select_button.clicked.connect(self.select_image)

        self.status_label = QLabel("Escolha uma imagem para rodar a deteccao.")
        self.status_label.setWordWrap(True)

        self.image_label = QLabel("Nenhuma imagem carregada.")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(800, 600)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.image_label)

        layout = QVBoxLayout()
        layout.addWidget(self.select_button)
        layout.addWidget(self.status_label)
        layout.addWidget(scroll_area)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def select_image(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Escolha a imagem", str(ROOT), IMAGE_FILTER)
        if not file_path:
            return

        try:
            annotated_image, summary = self.run_detection(Path(file_path))
        except Exception as exc:
            QMessageBox.critical(self, "Erro", str(exc))
            return

        self.status_label.setText(summary)
        self.current_pixmap = self.numpy_to_pixmap(annotated_image)
        self.update_image_preview()

    def run_detection(self, image_path: Path) -> tuple:
        results = self.model.predict(source=str(image_path), conf=0.25, verbose=False, device="cpu")
        result = results[0]

        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Nao foi possivel abrir a imagem: {image_path}")

        names = result.names
        detections = []

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            label = f"{names[cls_id]} {confidence:.2f}"
            detections.append(label)

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 0), 3)
            cv2.rectangle(image, (x1, max(0, y1 - 32)), (x2, y1), (0, 220, 0), -1)
            cv2.putText(
                image,
                label,
                (x1 + 6, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (20, 20, 20),
                2,
                cv2.LINE_AA,
            )

        if detections:
            summary = f"{image_path.name} | {len(detections)} deteccao(oes): " + ", ".join(detections)
        else:
            summary = f"{image_path.name} | nenhuma deteccao encontrada."

        return image, summary

    def numpy_to_pixmap(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_image.shape
        bytes_per_line = channels * width
        q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(q_image.copy())

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.update_image_preview()

    def update_image_preview(self) -> None:
        if self.current_pixmap is None:
            return

        scaled = self.current_pixmap.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)


def main() -> None:
    app = QApplication(sys.argv)
    window = DetectionWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
