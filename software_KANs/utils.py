import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from functions_2D import *

def scale_inputs_to_range(X_np, feature_range=(-0.9, 0.9)):
    scaler = MinMaxScaler(feature_range=feature_range)
    return scaler.fit_transform(X_np)

def prep_data(data):
    return torch.from_numpy(data).to(torch.float32).cpu()

def scatter_save_and_show(path, X2, colors, title, cmap="bwr", s=5, alpha=0.8, show=True, save=False):
    plt.figure(figsize=(4,4))
    plt.scatter(X2[:,0], X2[:,1], c=colors, cmap=cmap, s=s, alpha=alpha)
    # plt.xticks([]); plt.yticks([])
    plt.title(title)
    plt.tight_layout()
    if save:
        plt.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close()

def function_accuracy(model, dataset, threshold=0.05) -> None:
    # --- Calculate Training Accuracy ---
    train_predictions_raw = model(dataset['train_input'])
    train_true_labels = dataset['train_label']
    train_accuracy = (abs(train_predictions_raw - train_true_labels) <= threshold).float().mean()

    print(f"✅ Training Accuracy: {train_accuracy.item() * 100:.2f}%")


    # --- Calculate Test Accuracy (Recommended) ---
    test_predictions_raw = model(dataset['test_input'])
    test_true_labels = dataset['test_label']
    test_accuracy = (abs(test_predictions_raw - test_true_labels) <= threshold).float().mean()

    print(f"🧪 Test Accuracy: {test_accuracy.item() * 100:.2f}%")

def label_accuracy(model, dataset, threshold=0.5) -> None:
    # --- Calculate Training Accuracy ---
    train_predictions_raw = model(dataset['train_input'])
    train_predictions_labels = (train_predictions_raw > threshold).float()
    train_true_labels = dataset['train_label']
    train_true_labels  = (train_true_labels > threshold).float()
    train_accuracy = (train_true_labels == train_predictions_labels).float().mean()

    print(f"✅ Training Accuracy: {train_accuracy.item() * 100:.2f}%")

     # --- Calculate Test Accuracy (Recommended) ---
    test_predictions_raw = model(dataset['test_input'])
    test_predictions_labels = (test_predictions_raw > threshold).float().detach().numpy()
    test_true_labels = dataset['test_label']
    test_true_labels  = (test_true_labels > threshold).float()
    test_accuracy = (test_predictions_labels == test_true_labels).float().mean().detach().numpy()

    print(f"🧪 Test Accuracy: {test_accuracy.item() * 100:.2f}%")
    print(type(test_accuracy))
    return train_accuracy, test_accuracy

def save_task_data(path, task_name, train_acc, test_acc, results, params):
    data = {
        "task": f"{task_name.replace( '_', ' ')}",
        "train_acc": float(train_acc),
        "test_acc": float(test_acc),
        "train_loss": float(results["train_loss"][-1]),
        "test_loss": float(results["test_loss"][-1]),
        "width": params["model_size"],
        "grid": params["grid"],
        "seed": params["seed"],
        "test_split": params["test_split"],
        "steps": params["steps"],
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    return data

def print_split():
    print("\n" + "_"*50 + "\n")

def run_experiment(
    TEST_SPLIT: float,
    model_size: list[int],
    grid_size: int,
    seed: int,
    steps: int,
    threshold: int
):

    params = {
        "model_size": model_size,
        "grid": grid_size,
        "seed": seed,
        "test_split": TEST_SPLIT,
        "steps": steps
    }

    for index, task in enumerate(TASKS):
        print_split()
        name, func = task
        name = name.replace(" ", "_")
        print(f"Task {index+1}:\t{name}")

        ###########################################
        print("Prep data...")
        xs, label = func()
        xs = scale_inputs_to_range(xs)
        test_train_splits = train_test_split(xs, label, test_size=TEST_SPLIT)
        xs_train, xs_test, ys_train, ys_test = [prep_data(set) for set in test_train_splits]
        dataset = {
            "train_input": xs_train,
            "train_label": ys_train.unsqueeze(1)*7,
            "test_input": xs_test,
            "test_label": ys_test.unsqueeze(1)*7
        }

        ###########################################
        print("Initialise model")
        ###########################################
        model = KAN(width=[i for i in model_size], grid=grid_size, seed=seed)

        ###########################################
        print("Begin training...")
        ###########################################
        results = model.fit(dataset, steps=steps, lamb=0.01, lamb_entropy=10.0)

        ###########################################
        print("Results (plus saving):")
        ###########################################
        model_size_str = ""
        for i in model_size:
            model_size_str += f"{i}-"
        run_params = f"_g{grid_size}_m{model_size_str}"
        run = name + run_params
        model_path = "models/" + run + ".pth"
        fig_path = "data/figs/" + run
        data_path = "data/" + run + ".json"

        train_accuracy, test_accuracy = label_accuracy(model, dataset, threshold)

        scatter_save_and_show(fig_path+"_prediction", dataset['test_input'], model(dataset['test_input']).detach().numpy(), name + " pred", save=True)
        scatter_save_and_show(fig_path+"_threshold", dataset['test_input'], (model(dataset['test_input']).detach().numpy() >= threshold), name + " threshold", save=True)
        scatter_save_and_show(fig_path+"_actual", dataset['test_input'], dataset['test_label'], name + " ground truth", save=True)
        save_task_data(data_path, name, train_accuracy, test_accuracy, results, params)
        torch.save(model.state_dict(), model_path)