from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
from .doc_process import extract_text, chunk_text

apikey = "eNm1192jAWpKDnh7dkL-XB46y2-2otNAiASGGSGLyeYz"
project_id = "0e1ec565-5a03-4411-9046-06a2a89ae93d"
credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": f"{apikey}",
}
model_id = ModelTypes.GRANITE_13B_CHAT_V2
model = Model(model_id, credentials=credentials)


class Session:
    def __init__(self, model):
        history = []
        self.model = model
        self.instruction = """
        
        """

    def retrieval_info(self, question):
        pass

    def get_response(self, prompt):
        pass

def doc_summary(file_path):
    def extract_title_and_summary(text):
        lines = text.split('\n')
        title = lines[0].replace('Title: ', '').strip()
        summary = '\n'.join(lines[1:]).replace('Summary: ', '').strip()
        return title, summary
    text = extract_text(file_path)
    instruction = f"""
    You are an expert in summarizing documents and generating concise titles and summaries. Your task is to analyze the provided document and create a clear, brief title and summary that captures the main points of the content.

    Input: {text}
    Output:
    Title: [Generate a concise and relevant title based on the document's content]
    Summary: [Provide a clear and brief summary of the document, highlighting the key information and main ideas]
    """
    result = model.generate_text(instruction)
    title, summary = extract_title_and_summary(text)
    return title, summary

def chunk_summary(file_path):
    def get_prompt(text):
        return f"""
        You are an expert in summarizing documents and generating concise titles and summaries. Your task is to analyze the provided document and create a clear, brief title and summary that captures the main points of the content.
    
        Input: {text}
        Output:
        Summary: [Provide a clear and brief summary of the document, highlighting the key information and main ideas]
        """
    chunks = chunk_text(file_path)
    chunk_sums = [model.generate_text(get_prompt(chunk)).replace("Summary: ", "") for chunk in chunks]
    return chunk_sums