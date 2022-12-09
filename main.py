import pandas as pd
import time, os
import torch
from ntnmodel import LitModel
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

df = pd.DataFrame(columns=['old_state', 'new_state', 'reward'])
df2 = []
data = []
n = 4096
m = 1
def data_preprocessing(data):
    print("....data precessing....")
    z = []
    y = []
    X_data = [data[i][0] for i in range(len(data))]
    Y_data = [data[i][1] for i in range(len(data))]
    Z_data = [data[i][2] for i in range(len(data))]
    for i in range(len(data)):
        for j in range(len(data[i])):
            z.append([X_data[i][j], Y_data[i][j]])
    scale = StandardScaler().fit(z)
    z = scale.transform(z)
    norm = MinMaxScaler().fit(z)
    z = norm.transform(z)
    for i in range(len(data)):
        for j in range(len(data[i])):
            X_data[i][j], Y_data[i][j] = z[i+j][0], z[i+j][1]
    for i in range(len(Z_data)//2):
        y.append([Z_data[2*i], Z_data[2*i+1]])
    scale = StandardScaler().fit(y)
    y = scale.transform(y)
    norm = MinMaxScaler().fit(y)
    y = norm.transform(y)
    for i in range(len(Z_data)//2):
        Z_data[2*i], Z_data[2*i+1] = y[i][0], y[i][1]
    final_data = [[[torch.tensor(X_data[i], dtype= torch.float32), torch.tensor(Y_data[i], dtype= torch.float32)], torch.tensor(Z_data[i], dtype= torch.float32)] for i in range(len(data))]
    return final_data

checkpoints = ModelCheckpoint(dirpath= "/home/mord/tensor/tensor_model/", save_last= True , monitor='accuracy', 
                                        verbose=True, mode='max')
Model = LitModel().to("cuda")
while True:
    if not os.path.exists(f"data/dataset{n}.json"):
        pass
    else:
        df = pd.read_json(f"data/dataset{n}.json", dtype='float32', typ=['frame'], precise_float=True)
        df2 = df['data']
    if len(df2)==4096:
        print(f"dataset{n}")
        data = data_preprocessing(df2)
        train_loader =DataLoader(dataset=data[0: len(data)//2], pin_memory_device="cuda", pin_memory= True)
        val_loader =DataLoader(dataset=data[len(data)//2:len(data)], pin_memory_device="cuda", pin_memory= True)
        trainer = pl.Trainer(max_epochs=m, auto_lr_find=False, auto_scale_batch_size=False,
                                enable_progress_bar=True, callbacks=[checkpoints], accelerator="gpu")
        if not os.path.exists("/home/mord/tensor/tensor_model/"):
            trainer.fit(Model, train_dataloaders=train_loader)
            val = trainer.validate(Model, dataloaders= val_loader)
            

        else:
            trainer.fit(Model, train_dataloaders=train_loader, ckpt_path= "/home/mord/tensor/tensor_model/last.ckpt")
            val = trainer.validate(Model, dataloaders= val_loader, ckpt_path= "/home/mord/tensor/tensor_model/last.ckpt")
        print(val)
        acc = val[0]['accuracy']
        print(acc)
        if acc <0.9:
            pass
        else:
            trainer.save_checkpoint("/home/mord/tensor/deployed_tensor_model/last.ckpt")
        df2.clear()
        m = m +1
        n = n + 4096
    else:
        print(len(df2))
        time.sleep(0.5)
