import os, sys, requests, time, fire
from PIL import Image
from io import BytesIO


def mains(img_path, endpoint="https://westcentralus.api.cognitive.microsoft.com", key="5352d0636fc04db2923a94a6e11a3ce8", save_cred+False, saved_cred=True):
    '''
    A simply script to connect to your Azure Read API and convert img documents to text.
    
    Steps -
    1. Create a Azure credential from https://azure.microsoft.com/en-us/try/cognitive-services/?api=computer-vision
    2. Pass endpoint URL which looks like -> https://westcentralus.api.cognitive.microsoft.com to --endpoint flag.
    3. And padd api key ehich looks like -> 1234567890abcdefghijklmnopqrtsu (32 charecter long key) to --sub_key flag.
    
    NOTE: please encolse endpoint and key in a ""
    
    Flags
    --img_path   = (required) pass the local path of the image
    --endpoint   = Your Azure Cognitive Service Vision API endpoint
    --key        = Your Azure Cognitive Service Vision API key
    --save_cred  = Set True or False 
    
    '''
    text_recognition_url = endpoint + "/vision/v2.1/read/core/asyncBatchAnalyze"
    image_data = open(image_path, "rb").read()
    headers = {'Ocp-Apim-Subscription-Key': sub_key, 'Content-Type': 'application/octet-stream'}
    response = requests.post(text_recognition_url, headers=headers, data = image_data)
    response.raise_for_status()
    print(" Extracting text requires two API calls: One call to submit the image for processing, the other to retrieve the text found in the image. The recognized text isn't immediately available, so poll to wait for completion.")
    analysis = {}
    poll = True
    while (poll):
        analysis = (requests.get( response.headers["Operation-Location"], headers=headers)).json()
        time.sleep(1)
        if ("recognitionResults" in analysis): poll = False
        if ("status" in analysis and analysis['status'] == 'Failed'): poll = False
    text = []
    if ("recognitionResults" in analysis):
        text = [(line["text"]) for line in analysis["recognitionResults"][0]["lines"]]
    print(" ".join(text))
    
if __name__ == '__main__':
    fire.Fire(mains)
