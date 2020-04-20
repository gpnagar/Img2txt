import os, requests, time, fire
from io import BytesIO
from colorama import Fore

def mains(img_path="NaN", endpoint=None, key=None, cred_path=None, save_cred=None, txtout=None):
    '''
    A simply script to connect to your Microsoft Azure Read API and convert img documents to text. More like a wrapper API!
    
    Steps -
    1. Create a Azure credential from https://azure.microsoft.com/en-us/try/cognitive-services/?api=computer-vision
    2. Pass endpoint URL which looks like -> https://westcentralus.api.cognitive.microsoft.com to --endpoint flag.
    3. And pass api key which looks like -> 1234567890abcdefghijklmnopqrtsu (32 charecter long key) to --sub_key flag.
    4. You are required to pass the --img_path flag with a valid image path.
    
    NOTE: 
    * Please encolse endpoint and key in a "". 
    * Please wait after running the script cause it would take a min or 2 for the processing as the script makes two api calls one for sending the img and another for getting back the text!
    * 
    
    Flags
    --img_path   = (required) pass the local path of the image
    --endpoint   = Your Azure Cognitive Service Vision API endpoint
    --key        = Your Azure Cognitive Service Vision API key
    --save_cred  = Set True or False to save the Azure Cognitive Service Vision credentials
    --load_cred  = Load presaved key and endpoint from path
    --save_text  = Path to save the text file if not specified just return the text
    
    e.x.
    python .\AzureReadAPI.py --img_path="ImageDocument.jpeg" --endpoint="https://westcentralus.api.cognitive.microsoft.com" --key="1234567890qwertyuiopaassddd" --txtout=True --save_cred=True
    python .\AzureReadAPI.py --img_path="ImageDocument.jpeg"
    '''
    ##################### MY PERSONAL KEYS FOR TEST
#     endpoint="https://westcentralus.api.cognitive.microsoft.com"
#     key="5352d0636fc04db2923a94a6e11a3ce8"
    ###############################################################
    if save_cred:
        open('AZ_keys.key','w').write(endpoint+" "+key)
    if os.path.exists('AZ_keys.key'):
        kp = (open('AZ_keys.key','r').read()).split()
        endpoint, key = kp[0], kp[1]
    # Image works!
    if os.path.exists(img_path):
        image_data = open(img_path, "rb").read()
    else:
        print(Fore.RED + "Image not found!")
        return
    text_recognition_url = endpoint + "/vision/v2.1/read/core/asyncBatchAnalyze"
    headers = {'Ocp-Apim-Subscription-Key': key, 'Content-Type': 'application/octet-stream'}
    response = requests.post(text_recognition_url, headers=headers, data = image_data)
    response.raise_for_status()
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
        text = " ".join(text)
    if txtout:
        open('AZ_out.txt','w').write(text)
    else:
        return text
    
if __name__ == '__main__':
    fire.Fire(mains)
