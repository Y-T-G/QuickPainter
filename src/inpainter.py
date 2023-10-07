import cv2
import numpy as np
import openvino as ov


class Inpainter:
    def __init__(self, ir_path, device):
        # Load model
        core = ov.Core()
        model = core.read_model(model=ir_path)
        self.model = compiled_model = core.compile_model(
            model=model, device_name=device
        )

        # Store the input and output nodes
        self.input_layer = compiled_model.input(0)
        self.output_layer = compiled_model.output(0)

        # Shape NHWC
        N, H, W, C = self.input_layer.shape

        self.input_shape = (N, H, W, C)

    def apply_mask(self, image, masks, clip=0.1):
        # image: Original image
        # boxes: List of bounding boxes in (x, y, width, height) format
        # masks: List of binary masks

        full_mask = np.zeros(image.shape[:2]).astype(np.float32)

        # Overlay masks on bounding boxes and aggregate
        for mask in masks:
            # Clip
            mask = np.where(mask > clip, 1.0, 0.0).astype(np.float32)
            # Aggregate mask
            full_mask = cv2.bitwise_or(full_mask, mask)

            # Overlay the mask on the original image
            mask = mask[..., np.newaxis]
            image = (image * (1 - mask) + mask * 255.0).astype(np.uint8)

        return image, full_mask

    def preprocess(self, image, masks):

        prep_img, full_mask = self.apply_mask(image, masks)
        H, W = self.input_shape[1:3]

        prep_img = cv2.resize(prep_img, dsize=(W, H),
                              interpolation=cv2.INTER_AREA)
        prep_mask = cv2.resize(full_mask, dsize=(W, H),
                               interpolation=cv2.INTER_AREA)

        prep_img = prep_img[None, ...]
        prep_mask = prep_mask[None, ..., None]

        return prep_img, prep_mask

    def inpaint(self, image, masks):
        prep_img, prep_mask = self.preprocess(image, masks)
        result = self.model(
            [
                ov.Tensor(prep_img.astype(np.float32)),
                ov.Tensor(prep_mask.astype(np.float32)),
            ]
        )[self.output_layer]
        result = result.squeeze().astype(np.uint8)

        return result
