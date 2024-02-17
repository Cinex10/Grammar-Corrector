from typing import Any
import lightning.pytorch as pl
from transformers import AutoModelForSeq2SeqLM
from datasets import load_metric
#from nltk.translate.gleu_score import sentence_gleu

class T5Model(pl.LightningModule):
    def __init__(self,cfg):
        super(T5Model, self).__init__()
        self.save_hyperparameters()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model.name)
        self.gleu = load_metric("gleu","cola")
    
    def forward(self, input_ids,attention_mask,labels=None) -> Any:
        outputs = self.model(input_ids=input_ids,attention_mask=attention_mask, labels=labels)
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self.forward(
            batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"]
        )
        self.log("train/loss", outputs.loss, prog_bar=True, on_epoch=True)
        return outputs.loss

    def validation_step(self,batch,batch_idx):
        outputs = self.forward(
            batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"]
        )
        
        gleu_score = self.gleu.compute(predictions=outputs,references=batch["labels"])["matthews_correlation"]
        
        self.log("valid/loss", outputs.loss, prog_bar=True, on_epoch=True)
        self.log("valid/gleu", gleu_score, prog_bar=True, on_epoch=True)
        
        
    
        
    
    # def calculate_gleu(input : str, label : str):
    #     gleu_score = sentence_gleu([label.split()], input.split())
    #     return gleu_score
    
        