from .sikai_node import LoadAnyLLM, GenerateTextFromLLM, OpenAIDAlleNode, LoadNemotron, OpenAI_Text_Node, ToneLayerQuantize, ColorTransferToneLayer, ImageTracingNode, NaturalSaturationAdjust, greyscaleblendNode
from .sikai_text_tools import SKLoadText, SK_Save_Text, SK_Text_String, SK_Random_File_Name
NODE_CLASS_MAPPINGS = {
    "Load LLM": LoadAnyLLM,
    "Ask LLM": GenerateTextFromLLM,
    "SK Text_String": SK_Text_String,
    "OpenAI DAlle Node": OpenAIDAlleNode,
    "OpenAI Text Node": OpenAI_Text_Node,
    "SK Random File Name": SK_Random_File_Name,
    "Tone Layer Quantize": ToneLayerQuantize,
    "Color Transfer": ColorTransferToneLayer,
    "Image Tracing Node": ImageTracingNode,
    "Natural Saturation": NaturalSaturationAdjust,
    "grey_scale blend": greyscaleblendNode,
    "Load_Nemotron": LoadNemotron,
    "SK Save Text": SK_Save_Text,
    "SK load text": SKLoadText
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Load LLM": "Load AnyLLM",
    "Ask LLM": "Ask LLM",
    "OpenAI DAlle Node": "OpenAI DAlle Node",
    "OpenAI Text Node": "OpenAI Text Node",
    "SK Text_String": "SK Text_String",
    "SK Random File Name": "SK Random File Name",
    "Tone Layer Quantize": "Tone Layer Quantize",
    "Color Transfer": "Color Transfer",
    "Natural Saturation": "Natural Saturation",
    "Image Tracing Node": "Image Tracing Node",
    "grey_scale blend": "grey_scale blend",
    "Load_Nemotron": "Load_Nemotron",
    "SK Save Text": "SK Save Text",
    "SK load text": "SK load text"
}
__all__ = ['NODE_CLASS_MAPPINGS']