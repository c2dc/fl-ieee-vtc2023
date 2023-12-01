class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class Config:
    bsm = "23bsm"
    csv = f"../dismiss-bsm-vtc2023/models/vtc/dismiss/{bsm}/preprocessing/allmsg/allMsg-new-preds.csv"
    fedcsv = f"../dismiss-bsm-vtc2023/models/vtc/dismiss/{bsm}/preprocessing/allmsg/fed-test.csv"
    model_type = "mlp"
    label = "atk_2"
    feature = "feat4"
    batch_size = 200
    epochs = 5
    rounds = 350
    learning_rate = 1e-3
    min_available_clients = 2
    min_evaluate_clients = 2
    fraction_fit = 1
    output_activation = "softmax"
    early_stop_patience = 3
    early_stop_monitor = "loss"
    early_stop_min_delta = 1e-4
    early_stop_restore_best_weights = True
    data_train_size = 0.8
    data_test_size = 0.2
    performance_file = "performance.csv"
    weights_file = "model_weights.npz"


