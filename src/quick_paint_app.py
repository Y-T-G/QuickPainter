import numpy as np
import gradio as gr

from .inpainter import Inpainter
from .segmenter import Segmenter


class QuickPaintApp:
    def __init__(self, seg_ir_path, inp_ir_path, device="CPU"):
        self.segmenter = Segmenter(seg_ir_path, device)
        self.inpainter = Inpainter(inp_ir_path, device)
        self.app = gr.Blocks()

    # Function to apply inpainting based on selected checkboxes
    def inpaint_selected_objects(self, checkboxes, masks, class_ids, image_np):
        selected_labels = [selected for selected in checkboxes if checkboxes]
        selected_ids = [self.segmenter.label_map[label]
                        for label in selected_labels]
        filtered_masks = [
            mask for mask, cls_id in zip(masks, class_ids)
            if cls_id in selected_ids
        ]

        # Apply inpainting
        result = self.inpainter.inpaint(image_np, filtered_masks)

        return result

    # Gradio app configuration
    def segment(self, input_image):
        # Convert input_image to numpy array
        image_np = np.array(input_image)

        # Instance segmentation
        seg_out = self.segmenter.segment(image_np)
        boxes, masks, class_ids = (seg_out["boxes"], seg_out["masks"],
                                   seg_out["labels"])

        overlayed_img = self.segmenter.overlay_masks(image_np.copy(), masks)
        overlayed_img = self.segmenter.overlay_labels(
            overlayed_img, boxes[:, :4], class_ids, boxes[:, 4]
        )

        # Create checkboxes for each detected class
        detected_objects = [
            self.segmenter.labels[cls_id] for cls_id in np.unique(class_ids)
        ]
        checkboxes = gr.CheckboxGroup(
            label="Select objects to remove",
            choices=detected_objects,
            visible=True,
            interactive=True,
        )

        # Show inpaint button
        inpaint_button = gr.Button(value="Inpaint", visible=True)

        return (overlayed_img, checkboxes, inpaint_button, boxes, masks,
                class_ids)

    def build(self):
        with self.app:
            with gr.Row():
                with gr.Column():
                    input_img = gr.Image()
                    seg_btn = gr.Button(value="Segment")
                with gr.Column():
                    mask_img = gr.Image(label="Mask")
                    boxes = gr.State()
                    masks = gr.State()
                    class_ids = gr.State()
            with gr.Row():
                checkboxes = gr.CheckboxGroup(visible=False)
                inpaint_button = gr.Button(visible=False)
            with gr.Row():
                inpaint_canvas = gr.Image(label="Inpainted", visible=True)

            # Run segmentation on input image
            seg_btn.click(
                self.segment,
                inputs=[input_img],
                outputs=[mask_img, checkboxes, inpaint_button, boxes, masks,
                         class_ids],
                show_progress=True,
            )

            # # Inpaint button
            # inpaint_canvas = gr.Image(label="Inpainted", visible=False)
            inpaint_button.click(
                self.inpaint_selected_objects,
                inputs=[checkboxes, masks, class_ids, input_img],
                outputs=[inpaint_canvas],
                show_progress=True,
            )

    def launch(self):
        self.build()
        # Run Gradio app
        self.app.launch()

    def shutdown(self):
        self.app.close()
