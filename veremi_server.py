import logging
import os
import flwr as fl
import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
from config import Config
from itertools import cycle
from veremi_base import VeremiBase
from matplotlib import pyplot as plt
from flwr.common import NDArrays, Scalar
from flwr.common.logger import FLOWER_LOGGER
from veremi.veremi_fedavg import VeremiFedAvg
from typing import Optional, Tuple, Dict, Any
from sklearn.metrics import precision_recall_curve, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, auc
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, roc_curve
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

matplotlib.use("tkAgg")
tf.get_logger().setLevel('ERROR')

FLOWER_LOGGER.setLevel(logging.WARNING)


class VeremiServer(VeremiBase):
    def __init__(
            self,
            data_file: str,
            model_type: str,
            label: str,
            feature: str,
            rounds: int = 2,
            batch_size: int = 64,
            epochs: int = 10,
            activation: str = "softmax"
    ):
        super().__init__(data_file, model_type, label, feature, activation)
        self.rounds = rounds
        self.batch_size = batch_size
        self.epochs = epochs
        self.output_path = ""
        self.best_f1_score = 0
        self.output_path = f"results/{Config.bsm}/{feature}/{label}/"
        self.server_strategy = None
        self.f1_score_by_round = {}
        self.loss_by_round = {}

        self.load_data()

    def load_data(self):
        file = self.output_path + Config.performance_file
        if os.path.exists(file):
            df = pd.read_csv(file)
            if df is not None:
                self.best_f1_score = df['f1score'][0]

    def strategy(self):
        self.server_strategy = VeremiFedAvg(
            fraction_fit=Config.fraction_fit,
            min_available_clients=Config.min_available_clients,
            evaluate_fn=self.get_evaluate_fn(),
            on_fit_config_fn=self.get_config_fn(),
            output_path=self.output_path,
            min_evaluate_clients=Config.min_evaluate_clients,
        )
        return self.server_strategy

    def get_config_fn(self):
        def fit_config(server_round: int):
            return {
                "batch_size": self.batch_size,
                "epochs": self.epochs
            }

        return fit_config

    def get_evaluate_fn(self):
        def evaluate(server_round: int, weights: NDArrays, config: Dict[str, Scalar]) -> Optional[Tuple[float, Dict]]:
            if server_round < self.rounds - 1:
                return None

            self.model.set_weights(weights)
            loss, f1_score_result = self.model.evaluate(
                self.test_data,
                self.test_labels,
                verbose=0,
            )
            result = {
                "f1_score": float(f1_score_result),
            }

            print(f"=====> Round({str(server_round)}): loss: {float(loss):.4f} - f1: {float(f1_score_result):.4f} - "
                  f"Best f1: {float(self.best_f1_score):.4f}")
            self.f1_score_by_round[server_round] = float(f1_score_result)
            self.loss_by_round[server_round] = float(loss)
            if server_round == Config.rounds:
                self.save_round_loss_f1_score()

            if f1_score_result > self.best_f1_score:
                self.best_f1_score = f1_score_result
                self.save_model_and_plot_results(server_round)

            return loss, result

        return evaluate

    def save_round_loss_f1_score(self):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        rounds = self.f1_score_by_round.keys()
        rounds_f1_score = self.f1_score_by_round.values()
        rounds_loss = self.loss_by_round.values()

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Model Performance')
        fig.set_figwidth(15)
        fig.set_figheight(5)

        ax1.plot(rounds, rounds_loss, '-g', label="Loss")
        ax1.legend()
        ax1.set(xlabel='Round', ylabel='Loss')
        ax1.set_title('Loss')

        ax2.plot(rounds, rounds_f1_score, '-g', label="F1 Score")
        ax2.legend()
        ax2.set(xlabel='Round', ylabel='F1 Score')
        ax2.set_title('F1-Score')

        name = f"{self.label}-loss-f1score-{self.model_type}-{self.feature}"
        plt.savefig(f"{self.output_path}{name}.pdf")
        plt.close()

    def save_model_and_plot_results(self, server_round: int):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        self.server_strategy.save_params()
        self.model.save(self.output_path + self.model.name)

        print("Running prediction in " + self.__class__.__name__ + "...")
        probabilities = self.model.predict(self.test_data)
        inverse_target = self.lb.inverse_transform(self.test_labels)

        pos_label = 1
        pr_name = self.label + "-pr-curves-" + self.model_type + "-" + self.feature
        roc_name = self.label + "-roc-curves-" + self.model_type + "-" + self.feature
        if self.label == 'multiclass':
            prediction = self.lb.inverse_transform(probabilities)
            # PR and ROC Curves
            self.plot_roc_curves(roc_name, probabilities)
            self.plot_pr_curves(pr_name, probabilities)
        else:
            pos_label = 1 if self.label == 'binary' else int(self.label.split("_")[1])

            # Best threshold
            precision, recall, thresholds = precision_recall_curve(
                inverse_target,
                probabilities[:, 1],
                pos_label=pos_label
            )
            # convert to f score
            np.seterr(divide='ignore', invalid='ignore')
            fscore = (2 * precision * recall) / (precision + recall)
            np.nan_to_num(fscore, copy=False)
            # locate the index of the largest f score
            ix = np.argmax(fscore)
            print('Best Threshold=%f, F-Score=%.4f' % (thresholds[ix], fscore[ix]))
            print("-" * 70)
            # Classification Report
            prediction = np.where(np.array(probabilities[:, 1]) >= thresholds[ix], pos_label, 0)
            # ROC Curves
            self.plot_binary_pr_roc_curves(
                pr_name,
                roc_name,
                inverse_target,
                probabilities,
                pos_label,
                ix,
                precision,
                recall
            )
        # Classification Report
        report_name = self.label + "-report-" + self.model_type + "-" + self.feature
        self.print_classification_report(report_name, inverse_target, prediction)
        # Confusion matrix
        matrix_name = self.label + "-cfm-" + self.model_type + "-" + self.feature
        self.plot_confusion_matrix(matrix_name, inverse_target, prediction)
        # Performance
        perf_name = self.label + "-" + self.model_type + "-" + self.feature
        self.print_performance(perf_name, inverse_target, prediction, pos_label)

    def print_performance(self, name: str, target: Any, predictions: Any, pos_label: Any):
        if self.label == 'multiclass':
            prscore = precision_score(target, predictions, average='macro', zero_division=0)
            rcscore = recall_score(target, predictions, average='macro', zero_division=0)
            f1score = f1_score(target, predictions, average='macro', zero_division=0)
            accscore = accuracy_score(target, predictions)
        else:
            prscore = precision_score(target, predictions, pos_label=pos_label, zero_division=0)
            rcscore = recall_score(target, predictions, pos_label=pos_label, zero_division=0)
            f1score = f1_score(target, predictions, pos_label=pos_label, zero_division=0)
            accscore = accuracy_score(target, predictions)
        data_performance = {name: [prscore, rcscore, f1score, accscore]}
        df_performance = pd.DataFrame.from_dict(data_performance, orient='index',
                                                columns=["precision", "recall", "f1score", "accuracy"])
        fname = self.output_path + Config.performance_file
        try:
            performance = pd.read_csv(fname, index_col=0)
        except FileNotFoundError:
            df_performance.to_csv(fname)
        else:
            try:
                performance.loc[name] = df_performance.loc[name]
            except KeyError:
                performance = pd.concat([performance, df_performance])
            performance.to_csv(fname)

    def print_classification_report(self, name: str, target: Any, prediction: Any):
        classlist = []
        for cl in self.lb.classes_:
            classlist.append('class ' + str(int(cl)))
        with open(self.output_path + name + ".txt", "w") as f:
            f.write('Classification Report for ' + name + "\n")
            f.write(classification_report(target,
                                          prediction,
                                          target_names=classlist,
                                          digits=3,
                                          zero_division=0))
            f.write("-" * 70)
            f.flush()

    def plot_confusion_matrix(self, name: str, target: Any, prediction: Any):
        cm = confusion_matrix(target, prediction, labels=self.lb.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.lb.classes_)
        disp.plot()
        plt.title(name)
        plt.savefig(self.output_path + name + ".pdf")
        # plt.show()
        plt.close()

    def plot_roc_curves(self, name: str, probabilities: Any):
        n_classes = self.test_labels.shape[1]
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(self.test_labels[:, i], probabilities[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        lw = 2
        plt.figure()
        plt.plot(
            fpr["macro"],
            tpr["macro"],
            label="Macro Avg (area = {0:0.2f})".format(roc_auc["macro"]),
            color="navy",
            linestyle=":",
            linewidth=4,
            alpha=0.5,
        )

        colors = cycle(["b", "g", "r", "c", "m", "y"])
        for i, color in zip(range(n_classes), colors):
            label_classes = int(self.lb.classes_[i])
            plt.plot(
                fpr[i],
                tpr[i],
                color=color,
                lw=lw,
                label="Class {0} (area = {1:0.2f})".format(label_classes, roc_auc[i]),
                alpha=0.5
            )

        plt.plot([0, 1], [0, 1], lw=lw, color="grey", alpha=0.2)
        plt.xlim([-0.02, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.title("ROCcurve " + name)
        plt.savefig(self.output_path + name + ".pdf")
        # plt.show()
        plt.close()

    def plot_pr_curves(self, name: str, probabilities: Any):
        n_classes = self.test_labels.shape[1]
        precision = dict()
        recall = dict()
        pr_auc = dict()

        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(self.test_labels[:, i], probabilities[:, i])
            pr_auc[i] = auc(recall[i], precision[i])

        # First aggregate all false positive rates
        all_precision = np.unique(np.concatenate([precision[i] for i in range(n_classes)]))

        # Then interpolate all pr curves at this points
        mean_recall = np.zeros_like(all_precision)
        for i in range(n_classes):
            mean_recall += np.interp(all_precision, precision[i], recall[i])

        # Finally average it and compute AUC
        mean_recall /= n_classes

        precision["macro"] = all_precision
        recall["macro"] = mean_recall
        pr_auc["macro"] = auc(recall["macro"], precision["macro"])

        # Plot all pr curves
        lw = 2
        plt.figure()
        plt.plot(
            precision["macro"],
            recall["macro"],
            label="Macro Avg (area = {0:0.2f})".format(pr_auc["macro"]),
            color="navy",
            linestyle=":",
            linewidth=4,
            alpha=0.5,
        )

        colors = cycle(["b", "g", "r", "c", "m", "y"])
        for i, color in zip(range(n_classes), colors):
            label_classes = int(self.lb.classes_[i])
            plt.plot(
                precision[i],
                recall[i],
                color=color,
                lw=lw,
                label="Class {0} (area = {1:0.2f})".format(label_classes, pr_auc[i]),
                alpha=0.5
            )

        plt.xlim([0.15, 1.02])
        plt.ylim([-0.01, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower left")
        plt.title("PRcurve " + name)
        plt.savefig(self.output_path + name + ".pdf")
        # plt.show()
        plt.close()

    def plot_binary_pr_roc_curves(
            self,
            prname: str,
            rocname: str,
            target: Any,
            probabilities: Any,
            pos_label: Any,
            ix: Any,
            precision: Any,
            recall: Any
    ):
        PrecisionRecallDisplay.from_predictions(target, probabilities[:, 1], pos_label=pos_label)
        plt.title('PR curve ' + prname)
        no_skill = len(target[target == 1]) / len(target)
        plt.plot([0, 1], [no_skill, no_skill], linestyle='--', color="grey", label='No Skill')
        plt.scatter(recall[ix], precision[ix], marker='o', color='black', label='Best threshold')
        plt.legend()
        plt.savefig(self.output_path + prname + '.pdf')
        plt.close()

        # ROC curve
        RocCurveDisplay.from_predictions(target, probabilities[:, 1], pos_label=pos_label)
        plt.title('ROC curve ' + rocname)
        plt.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--")
        plt.savefig(self.output_path + rocname + ".pdf")
        plt.close()


if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    server = VeremiServer(
        data_file=Config.csv,
        model_type=Config.model_type,
        label=Config.label,
        feature=Config.feature,
        batch_size=Config.batch_size,
        epochs=Config.epochs,
        activation=Config.output_activation
    )

    # Start the Flower Server
    fl.server.start_server(
        config=fl.server.ServerConfig(num_rounds=Config.rounds),
        strategy=server.strategy()
    )
