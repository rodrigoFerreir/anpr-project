import easyocr


class EasyOCRAdapter:
    """
    EasyOCRAdapter is a wrapper around the EasyOCR library to perform OCR on images.
    """

    def __init__(self, lang_list=None):
        """
        Initialize the EasyOCRAdapter with the specified language list.

        :param lang_list: List of languages to use for OCR. If None, defaults to ['en'].
        """
        lang_list = lang_list or ["en"]
        self.reader = easyocr.Reader(lang_list, gpu=True)

    def readtext(self, image):
        """
        Perform OCR on the given image.

        :param image: The image to perform OCR on.
        :return: A list of detected text and their bounding boxes.
        """
        return self.reader.readtext(image)[0]
