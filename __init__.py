from .sikai_node import imageselect, LoadAnyLLM, GenerateTextFromLLM, OpenAIDAlleNode, LoadNemotron, SK_Text_String, SK_Random_File_Name, OpenAI_Text_Node, ToneLayerQuantize, ColorTransferToneLayer, ImageTracingNode, NaturalSaturationAdjust, greyscaleblendNode

NODE_CLASS_MAPPINGS = {
    "SK Text_String": SK_Text_String,
    "OpenAI Text Node": OpenAI_Text_Node,
    "SK Random File Name": SK_Random_File_Name,
    "Tone Layer Quantize": ToneLayerQuantize,
    "Image Tracing Node": ImageTracingNode,
    "Natural Saturation": NaturalSaturationAdjust,
    "grey_scale blend": greyscaleblendNode,
    "Load_Nemotron": LoadNemotron
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenAI Text Node": "OpenAI Text Node",
    "SK Text_String": "SK Text_String",
    "SK Random File Name": "SK Random File Name",
    "Tone Layer Quantize": "Tone Layer Quantize",
    "Natural Saturation": "Natural Saturation",
    "Image Tracing Node": "Image Tracing Node",
    "grey_scale blend": "grey_scale blend",
    "Load_Nemotron": "Load_Nemotron"
}
__all__ = ['NODE_CLASS_MAPPINGS']
