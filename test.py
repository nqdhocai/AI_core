from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models import Model



apikey = "eNm1192jAWpKDnh7dkL-XB46y2-2otNAiASGGSGLyeYz"
project_id = "0e1ec565-5a03-4411-9046-06a2a89ae93d"

credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": f"{apikey}",
}

model_id = ModelTypes.GRANITE_13B_CHAT_V2

parameters = {
    GenParams.DECODING_METHOD: "greedy",
    GenParams.MAX_NEW_TOKENS: 100,
    GenParams.STOP_SEQUENCES: ["\n\n"]
}

model = Model(
    model_id=model_id,
    params=parameters,
    credentials=credentials,
    project_id=project_id)
t = "RL là học cái để thực hiện, tức là từ các tình huống thực tế để đưa ra các action nhất định, miễn là maximize được reward. Machine không được bảo về cái action để thực hiện mà thay vào đó phải khám phá ra action có thể tạo ra được nhiều reward nhất. Trong thế giới của RL thì chúng ta có khái niệm gọi là agent, nó có một chút gì đó hàm ý về một thực thể mà bạn mong muốn train nó để có thể làm được một task nào đó mà bạn giao phó (đương nhiên là nó sẽ thực hiện theo cách đạt được reward nhiều nhất)."
instruction = """
    You are an expert in content categorization, specializing in identifying broad topics and major fields of study. Your task is to analyze the provided document and generate tags that represent the primary categories or major fields relevant to the document. Please provide the output in JSON format with the following structure:

 {
  "tags": [
    "Tag 1",
    "Tag 2",
    "Tag 3"
  ]
}

Input: RL là học cái để thực hiện, tức là từ các tình huống thực tế để đưa ra các action nhất định, miễn là maximize được reward. Machine không được bảo về cái action để thực hiện mà thay vào đó phải khám phá ra action có thể tạo ra được nhiều reward nhất. Trong thế giới của RL thì chúng ta có khái niệm gọi là agent, nó có một chút gì đó hàm ý về một thực thể mà bạn mong muốn train nó để có thể làm được một task nào đó mà bạn giao phó (đương nhiên là nó sẽ thực hiện theo cách đạt được reward nhiều nhất).
Output: [Generate a list of broad topic tags that describe the major fields of the document, using the format shown above]
    """

result = model.generate_text(instruction)
print(result)
