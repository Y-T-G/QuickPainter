from pathlib import Path

from src.quick_paint_app import QuickPaintApp
from utils.utils import download_file, download_ir_model
from zipfile import ZipFile

import openvino as ov


if __name__ == "__main__":
    base_model_dir = "model"

    # Download Instance Segmentation model
    model_name = "instance-segmentation-security-1040"

    model_path = Path(f"{base_model_dir}/{model_name}.xml")
    if not model_path.exists():
        model_xml_url = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/temp/instance-segmentation-security-1040/FP16/instance-segmentation-security-1040.xml"
        download_ir_model(model_xml_url, base_model_dir)

    # Download inpainter model
    model_name = "gmcnn-places2-tf"

    model_path = Path(f"{base_model_dir}/public/{model_name}/frozen_model.pb")
    if not model_path.exists():
        model_url = f"https://storage.openvinotoolkit.org/repositories/open_model_zoo/public/2022.1/gmcnn-places2-tf/{model_name}.zip"
        download_file(model_url, model_name, base_model_dir)

        with ZipFile(f"{base_model_dir}/{model_name}" + "", "r") as zip_ref:
            zip_ref.extractall(
                path=Path(
                    base_model_dir,
                    "public",
                )
            )

    # Convert inpainter model
    model_dir = Path(base_model_dir)
    ir_path = Path(f"{model_dir}/{model_name}.xml")
    if not ir_path.exists():
        # Run model conversion API to convert model to OpenVINO IR FP32 format,
        # if the IR file does not exist.
        ov_model = ov.convert_model(
            model_path, input=[[1, 512, 680, 3], [1, 512, 680, 1]]
        )
        ov.save_model(ov_model, str(ir_path))

    app = QuickPaintApp(
        "model/instance-segmentation-security-1040.xml",
        "model/gmcnn-places2-tf.xml",
    )

    app.launch()
