import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from src.model import MultiOmicsNet
from src.data_preprocessing import load_data, normalize_data, align_data

# CONFIG
omics_paths = {
    "rna": "data/processed/rna.csv",
    "meth": "data/processed/methylation.csv",
    "cnv": "data/processed/cnv.csv"
}
label_path = "data/processed/labels.csv"

def prepare_dataset(data_dict, labels):
    encoder = LabelEncoder()
    y = torch.tensor(encoder.fit_transform(labels)).long()

    tensors = {
        omic: torch.tensor(df.T.values, dtype=torch.float32)
        for omic, df in data_dict.items()
    }

    return tensors, y, encoder

def main():
    raw_data = load_data(omics_paths)
    norm_data = normalize_data(raw_data)
    aligned_data, y = align_data(norm_data, label_path)
    x_dict, y_tensor, encoder = prepare_dataset(aligned_data, y)

    train_idx, test_idx = train_test_split(np.arange(len(y_tensor)), test_size=0.2, stratify=y_tensor)
    train_loader = DataLoader(TensorDataset(
        *(x[train_idx] for x in x_dict.values()), y_tensor[train_idx]
    ), batch_size=32, shuffle=True)

    test_loader = DataLoader(TensorDataset(
        *(x[test_idx] for x in x_dict.values()), y_tensor[test_idx]
    ), batch_size=32)

    input_dims = {omic: x.shape[1] for omic, x in x_dict.items()}
    model = MultiOmicsNet(input_dims=input_dims)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(20):
        model.train()
        epoch_loss = 0
        for *inputs, labels in train_loader:
            inputs_dict = {k: v for k, v in zip(x_dict.keys(), inputs)}
            optimizer.zero_grad()
            outputs = model(inputs_dict)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), "models/deepmo_bc.pth")

if __name__ == "__main__":
    main()
