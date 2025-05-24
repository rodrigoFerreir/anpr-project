import cv2
import re
import logging

from ultralytics.utils.plotting import Annotator, colors

logger = logging.getLogger(__name__)


class Inference:
    def __init__(self, model_anpr, reader):
        """
        Inicializa o modelo de ANPR (Reconhecimento Automático de Placas de Veículos) e o leitor OCR.
        :param model_path: Caminho para o modelo YOLOv11 (ex: 'anpr-demo-model.pt')
        :param reader: Instância do leitor OCR (ex: easyocr.Reader)
        """
        self.model_anpr = model_anpr
        self.reader = reader

    # reader = PaddleOCR(use_angle_cls=True, lang="en")

    def inference_ocr(self, image):
        try:
            if self.reader is None:
                raise ValueError("O leitor OCR não foi inicializado.")

            _, plate, accuracy = self.reader.readtext(image)

            logger.info(f"OCR Result: {plate}, Confidence: {accuracy}")

            return str(plate), "{:.2f}".format(float(accuracy))
        except:
            return "", "0"

    def inference(self, frame: cv2.typing.MatLike):
        padding = 10
        results = self.model_anpr.track(frame)[0].boxes
        boxes = results.xyxy.cpu()
        clss = results.cls.cpu()

        ann = Annotator(frame, line_width=2)

        for (
            clss,
            box,
        ) in zip(clss, boxes):
            height, width, _ = frame.shape  # Get the dimensions of the original image
            class_name = self.model_anpr.class_name(int(clss))

            # Calculate padded coordinates
            x1 = max(int(box[0]) - padding, 0)
            y1 = max(int(box[1]) - padding, 0)
            x2 = min(int(box[2]) + padding, width)
            y2 = min(int(box[3]) + padding, height)

            # Crop the object with padding and encode the numpy array to base64 format.
            image_to_OCR = frame[y1:y2, x1:x2]

            image_to_OCR = cv2.cvtColor(image_to_OCR, cv2.COLOR_BGR2GRAY)

            try:
                result_OCR, accuracy = self.inference_ocr(image_to_OCR)

                result = re.sub(r"[^A-Z0-9]", "", result_OCR)
                logger.info(result)

                ann.box_label(
                    box,
                    label=f"{class_name} - {result} - {accuracy}",
                    color=colors(clss, True),
                )  # Draw the bounding boxes
            except Exception as e:
                logger.error(f"Error processing OCR: {e}")

        return frame
