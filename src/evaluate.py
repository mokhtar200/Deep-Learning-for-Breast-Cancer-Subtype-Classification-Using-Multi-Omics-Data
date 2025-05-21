import torch
from sklearn.metrics import classification_report, confusion_matrix
from src.model import MultiOmicsNet
from src.train import prepare_dataset, omics_paths, label_path
from src.data_preprocessing import load_data, normalize_data, align_data

def evaluate():
    raw_data = load_data(omics_paths)
    norm_data = normalize_data(raw_data)
    aligned_data, y = align_data(norm_data, label_path)
    x_dict, y_tensor, encoder = prepare_dataset(aligned_data, y)

    input_dims = {omic: x.shape[1] for omic, x in x_dict.items()}
    model = MultiOmicsNet(input_dims)
    model.load_state_dict(torch.load("models/deepmo_bc.pth"))
    model.eval()

    with torch.no_grad():
        inputs_dict = {omic: x for omic, x in x_dict.items()}
        outputs = model(inputs_dict)
        preds = torch.argmax(outputs, dim=1)

    print("Classification Report:")
    print(classification_report(y_tensor, preds, target_names=encoder.classes_))
    print("Confusion Matrix:")
    print(confusion_matrix(y_tensor, preds))

if __name__ == "__main__":
    evaluate()
