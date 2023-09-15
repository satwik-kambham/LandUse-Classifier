import gradio as gr
import numpy as np
import onnxruntime as ort
import torchvision as tv
from PIL import Image
from huggingface_hub import hf_hub_download

CATEGORIES = [
    "agricultural",
    "airplane",
    "baseballdiamond",
    "beach",
    "buildings",
    "chaparral",
    "denseresidential",
    "forest",
    "freeway",
    "golfcourse",
    "harbor",
    "intersection",
    "mediumresidential",
    "mobilehomepark",
    "overpass",
    "parkinglot",
    "river",
    "runway",
    "sparseresidential",
    "storagetanks",
    "tenniscourt",
]


class Classifier:
    def __init__(self, model_path):
        self.model_path = model_path
        self.session = ort.InferenceSession(self.model_path)

        self.img_transforms = tv.transforms.Compose(
            [
                tv.transforms.Resize((256, 256)),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(
                    (0.48422758, 0.49005175, 0.45050276),
                    (0.17348297, 0.16352356, 0.15547496),
                ),
            ]
        )

    def predict(self, image):
        inp = self.img_transforms(image).unsqueeze(0).numpy()
        logits = self.session.run(
            None,
            {self.session.get_inputs()[0].name: inp},
        )[0]
        probs = np.exp(logits) / np.sum(np.exp(logits))
        return {category: float(prob) for category, prob in zip(CATEGORIES, probs[0])}


model_path = hf_hub_download(
    repo_id="SatwikKambham/land_use_classifier",
    filename="model.onnx",
)
classifier = Classifier(model_path)
interface = gr.Interface(
    fn=classifier.predict,
    inputs=gr.components.Image(label="Input image", type="pil"),
    outputs=gr.components.Label(label="Predicted class", num_top_classes=3),
)
interface.launch()
