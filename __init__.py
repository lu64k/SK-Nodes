from .sikai_node import imageselect, LoadAnyLLM, GenerateTextFromLLM, OpenAIDAlleNode, SK_Text_String, SK_Random_File_Name

NODE_CLASS_MAPPINGS = {
    "Image Select": imageselect,
    "Load LLM": LoadAnyLLM,
    "Ask LLM": GenerateTextFromLLM,
    "SK Text_String": SK_Text_String,
    "OpenAI DAlle Node": OpenAIDAlleNode,
    "SK Random File Name": SK_Random_File_Name
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Image Select": "image select",
    "Load LLM": "Load AnyLLM",
    "Ask LLM": "Ask LLM",
    "OpenAI DAlle Node": "OpenAI DAlle Node",
    "SK Text_String": "SK Text_String",
    "SK Random File Name": "SK Random File Name"
}
__all__ = ['NODE_CLASS_MAPPINGS']