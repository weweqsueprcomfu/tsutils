from .nodes.save_load_pose import TSSavePoseDataAsPickle, TSLoadPoseDataPickle
from .nodes.openpose_smoother import KPSSmoothPoseDataAndRender
from .nodes.load_video_batch import LoadVideoBatchListFromDir
from .nodes.rename_files import RenameFilesInDir
from .nodes.color_match import TSColorMatchSequentialBias
from .nodes.preview_image_metadata import PreviewImageNoMetadata
from .nodes.video_combine_metadata import TSVideoCombineNoMetadata
from .nodes.upscaler import TSUpscaler
from .nodes.downscaler import TSDownscaler
from .nodes.denoise import TSDenoise

NODE_CLASS_MAPPINGS = {
    "TSSavePoseDataAsPickle": TSSavePoseDataAsPickle,
    "TSLoadPoseDataPickle": TSLoadPoseDataPickle,
    "TSPoseDataSmoother": KPSSmoothPoseDataAndRender,
    "TSLoadVideoBatchListFromDir": LoadVideoBatchListFromDir,
    "TSRenameFilesInDir": RenameFilesInDir,
    "TSColorMatch": TSColorMatchSequentialBias,
    "TSPreviewImageNoMetadata": PreviewImageNoMetadata,
    "TSVideoCombineNoMetadata": TSVideoCombineNoMetadata,
    "TSUpscaler": TSUpscaler,
    "TSDownscaler": TSDownscaler,
    "TSDenoise": TSDenoise,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TSSavePoseDataAsPickle": "TS Save Pose Data (PKL)",
    "TSLoadPoseDataPickle": "TS Load Pose Data (PKL)",
    "TSPoseDataSmoother": "TS Pose Data Smoother",
    "TSLoadVideoBatchListFromDir": "TS Load Video Batch List From Dir",
    "TSRenameFilesInDir": "TS Rename Files In Dir",
    "TSColorMatch": "TS Color Match",
    "TSPreviewImageNoMetadata": "TS Preview Image No Metadata",
    "TSVideoCombineNoMetadata": "TS Video Combine No Metadata",
    "TSUpscaler": "TS Upscaler",
    "TSDownscaler": "TS Downscaler",
    "TSDenoise": "TS Denoise",
}

WEB_DIRECTORY = "web"
