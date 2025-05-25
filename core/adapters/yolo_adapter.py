from ultralytics import YOLO
from ultralytics.utils import ThreadingLocked


class YoloAdapter:
    def __init__(self, model_name: str):
        self.model = YOLO(model_name)

    def class_name(self, class_id: int):
        # Get the class name from the model
        return self.model.names[class_id]

    @ThreadingLocked()
    def predict(self, image):
        # Perform prediction using the YOLO model
        return self.model.predict(image)

    @ThreadingLocked()
    def track(self, image):
        # Perform tracking using the YOLO model
        return self.model.track(image)
