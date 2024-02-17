import lightning.pytorch as pl
from torch.utils.data import DataLoader
from transformers import  T5Tokenizer
from datasets import load_dataset


class C4200DatasetModule(pl.LightningDataModule):
    def __init__(self,cfg):
        self.batch_size = cfg.processing.batch_size
        self.num_worker = cfg.processing.num_worker
        self.dataset_name = cfg.training.dataset_name
        self.tokenizer = T5Tokenizer.from_pretrained(cfg.model.name)
        self.val_set_size = cfg.training.validation_set_size
        self.max_length = cfg.processing.max_length
        self.prefix = cfg.processing.prefix
        
        
    def prepare_data(self):
        c4_200_dataset = load_dataset(self.dataset_name)
        data = c4_200_dataset['train']
        data = data.train_test_split(test_size=self.val_set_size)
        self.train_data,self.val_data = data['train'],data['test']
    
    def preprocess_data(self,examples):
        inputs = [self.prefix + doc for doc in examples['input']]
        model_inputs = self.tokenizer(inputs,
                                      max_length=self.max_length,
                                      truncation=True,
                                      padding="max_length",
                                      return_tensors='pt'
                                      )
        labels = self.tokenizer(examples['output'],
                                max_length=self.max_length,
                                truncation=True,
                                padding="max_length",
                                return_tensors='pt'
                                )
        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    
    def setup(self, stage: str):
        if stage == 'fit':
            self.train_data = self.train_data.map(self.preprocess_data,batched=True)
        if stage == 'validate':
            self.val_data = self.val_data.map(self.preprocess_data,batched=True)
    
    def train_dataloader(self):
        return DataLoader(self.train_data,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_worker
                          )
        
    def val_dataloader(self):
        return DataLoader(self.train_data,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_worker
                          )
        
        
    
    


