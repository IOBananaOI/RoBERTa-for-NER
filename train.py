import os

import torch
import torch.nn.functional as F
import neptune

from tqdm import tqdm 
from torchmetrics import Accuracy, F1Score

import numpy as np


def get_neptune_config(neptune_run_id=None):
    return {
        'neptune_project_name' : 'bng215/Model-Collapse',
        'neptune_project_api_token' : os.environ.get('NEPTUNE_API_TOKEN'),
        'neptune_run_id' : neptune_run_id
    }

def train(model, optimizer, num_epochs,
          train_loader, val_loader, start_epoch=0):
    
    npt_config = get_neptune_config()

    neptune_run = neptune.init_run(
        project=npt_config['neptune_project_name'],
        with_id=npt_config['neptune_run_id'],
        api_token=npt_config['neptune_project_api_token']
        )
    
    f1 = F1Score(task="multiclass", num_classes=9)
    
    for epoch in range(start_epoch, num_epochs):
        with tqdm(train_loader) as tepoch:
            model.train()
            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}: Train phase")
                optimizer.zero_grad()
                
                tokens, tags = batch
                
                tokens = torch.stack(tokens, dim=1).cuda()
                tags = torch.stack(tags, dim=1).cuda()
                attention_mask = (tokens != 1).to(torch.uint8)
                
                model_results = model(tokens, labels=tags, attention_mask=attention_mask)
                
                loss = model_results.loss
                logits = model_results.logits
                
                loss.backward()
                optimizer.step()
                
                # Track
                neptune_run['Loss (Train)'].append(loss.item())
            
        test_loss = []
        test_accs = []
        
        with tqdm(val_loader) as tepoch:
            model.eval()
            with torch.no_grad():
                for batch in tepoch:
                    tepoch.set_description(f"Epoch {epoch + 1}: Test phase")
                    
                    tokens, tags = batch
                    
                    tokens = torch.stack(tokens, dim=1).cuda()
                    tags = torch.stack(tags, dim=1).cuda()
                    attention_mask = (tokens != 1).to(torch.uint8)
                    
                    model_results = model(tokens, labels=tags, attention_mask=attention_mask)
                    
                    loss = model_results.loss
                    logits = model_results.logits
                    
                    pred = F.softmax(logits, dim=-1).argmax(dim=-1).cuda().squeeze()
                    
                    y_true = tags[tags != -100].squeeze()
                    
                    if type(y_true.tolist()) == int:
                        y_true = [y_true]

                    pred = pred[1:len(y_true) + 1]
                    
                    if type(y_true) == list:
                        y_true = torch.tensor(y_true)
                    
                    test_loss.append(loss.item())
                    test_accs.append(f1(pred.cpu(), y_true.cpu()))
                    
                    # Track
        neptune_run['Loss (Test)'].append(np.mean(test_loss))
        neptune_run['F1 (Test)'].append(np.mean(test_accs))
        
        torch.save(model.state_dict(), f"weights/ep_{epoch+1}.pth")
            
                