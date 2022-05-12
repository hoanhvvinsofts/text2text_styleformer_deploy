from django.http import HttpResponse
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated

import time
import json
import warnings
warnings.filterwarnings("ignore")

class Styleformer():
  def __init__(self,  style=1):
    from transformers import AutoTokenizer
    from transformers import AutoModelForSeq2SeqLM

    self.style = style
    self.model_loaded = False

    if self.style == 1:
      self.ftc_tokenizer = AutoTokenizer.from_pretrained("./api/formal_to_informal_styletransfer/", local_files_only=True)
      self.ftc_model = AutoModelForSeq2SeqLM.from_pretrained("./api/formal_to_informal_styletransfer/", local_files_only=True)
      print("Formal to Casual model loaded...")
      self.model_loaded = True
    else:
      print(">> THIS MODULE WORK ONLY WITH STYLE = 1: Formal to Casual")

  def transfer(self, input_sentence, inference_on=-1, quality_filter=0.95, max_candidates=5):
      if self.model_loaded:
        if inference_on == -1:
          device = "cpu"
        elif inference_on >= 0 and inference_on < 999:
          device = "cuda:" + str(inference_on)
        else:  
          device = "cpu"
          print("Onnx + Quantisation is not supported in the pre-release...stay tuned.")

        if self.style == 1:
          output_sentence = self._formal_to_casual(input_sentence, device, quality_filter, max_candidates)
          return output_sentence 
      else:
        print("Models aren't loaded for this style, please use the right style during init")  

  def _formal_to_casual(self, input_sentence, device, quality_filter, max_candidates):
      ftc_prefix = "transfer Formal to Casual: "
      input_sentence = ftc_prefix + input_sentence
      input_ids = self.ftc_tokenizer.encode(input_sentence, return_tensors='pt')
      self.ftc_model = self.ftc_model.to(device)
      input_ids = input_ids.to(device)
      
      preds = self.ftc_model.generate(
          input_ids,
          do_sample=True, 
          max_length=32, 
          top_k=50, 
          top_p=0.95, 
          early_stopping=True,
          num_return_sequences=max_candidates)
     
      gen_sentences = set()
      for pred in preds:
        gen_sentences.add(self.ftc_tokenizer.decode(pred, skip_special_tokens=True).strip())

      return list(gen_sentences)[-1]
  
sf = Styleformer(style = 1)

def home(request):
    return HttpResponse("<html><body>WORKING. . .</body></html>")

class ServiceText2Text(APIView):
    permission_classes = (IsAuthenticated,)
    @staticmethod
    def get(request, input_sentences="But Earth is Flat and space is hoax and earth has dome. Where does these rockets go ?"):
        '''
        Example:
                begun and began are not the same word.. this announcer is not great.. and why was this announced mission completed?
            But Earth is Flat and space is hoax and earth has dome. Where does these rockets go ?
            Remember folks, this is coming from a man who was born with all his basic needs met and more, talking down on people who work for what they got.
        '''
        
        input_sentences = request.headers['input']
        # inference_on = [-1=Regular model On CPU, 0-998= Regular model On GPU, 999=Quantized model On CPU]
        target_sentence = sf.transfer(input_sentences, inference_on=-1, quality_filter=0.95, max_candidates=2)
        
        if target_sentence is not None:
            return Response({"output": target_sentence}, status=200)
        else:
            return Response({"output": None})
    