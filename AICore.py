from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
from .doc_process import extract_text, chunk_text
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ast import literal_eval

apikey = "eNm1192jAWpKDnh7dkL-XB46y2-2otNAiASGGSGLyeYz"
project_id = "0e1ec565-5a03-4411-9046-06a2a89ae93d"
credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": f"{apikey}",
}
parameters = {
    GenParams.DECODING_METHOD: "greedy",
    GenParams.MAX_NEW_TOKENS: 1000,
    GenParams.STOP_SEQUENCES: ["\n\n"]
}
model_id = ModelTypes.GRANITE_13B_CHAT_V2
model = Model(
    model_id=model_id,
    params=parameters,
    credentials=credentials,
    project_id=project_id
)


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
    text = extract_text(file_path)
    instruction = """
    You are an expert in summarizing documents and generating concise titles and summaries. Your task is to analyze the provided document and create a clear, brief title and summary that captures the main points of the content.Please provide the output in JSON format with the following structure:
    {"title": title, "summary": summary}
    """+f""""
    Input: {text}
    Output:
    Title: [Generate a concise and relevant title based on the document's content]
    Summary: [Provide a clear and brief summary of the document, highlighting the key information and main ideas]
    """
    result = literal_eval(model.generate_text(instruction))
    title, summary = result['title'], result['summary']
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


def get_tag(summary):
    def get_prompt(text):
        instruction = """
        You are an expert in content categorization, specializing in identifying broad topics and major fields of study. Your task is to analyze the provided document and generate tags that represent the primary categories or major fields relevant to the document. Please provide the output in JSON format with the following structure:
    
        {"tags": ["Tag 1","Tag 2","Tag 3"]}
        """ + f"""
        Input: {text}
        Output: [Generate a list of broad topic tags that describe the major fields of the document, using the format shown above]
        """
        return instruction
    result = literal_eval(model.generate_text(get_prompt(summary)))
    return result['tags']