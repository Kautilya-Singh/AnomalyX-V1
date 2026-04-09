"""
Run this once before starting the web app:
    python train.py --train KDDTrain+.txt --test KDDTest+.txt
"""
import argparse
from network_anomaly_detector import *

def main():
    parser = argparse.ArgumentParser(description='Train the Network Anomaly Detector')
    parser.add_argument('--train', required=True, help='Path to KDDTrain+.txt')
    parser.add_argument('--test', required=True, help='Path to KDDTest+.txt')
    args = parser.parse_args()

    print("=" * 55)
    print("  NETWORK ANOMALY DETECTOR — TRAINING PIPELINE")
    print("=" * 55)

    print("\n[1/4] Loading & preprocessing training data...")
    X_train, y_train, encoders, scaler = load_and_preprocess(args.train, fit=True)
    print(f"      ✓ {len(y_train)} samples loaded")

    print("\n[2/4] Training Random Forest (100 trees, all CPU cores)...")
    model = train_model(X_train, y_train)
    print("      ✓ Model trained")

    print("\n[3/4] Evaluating on test set...")
    X_test, y_test, _, _ = load_and_preprocess(args.test, encoders=encoders, scaler=scaler, fit=False)
    y_pred, acc, report, cm = evaluate_model(model, X_test, y_test)

    print(f"\n      ✅ Accuracy : {acc * 100:.2f}%")
    print(f"      Precision (Attack) : {report['Attack']['precision']:.4f}")
    print(f"      Recall    (Attack) : {report['Attack']['recall']:.4f}")
    print(f"      F1-Score  (Attack) : {report['Attack']['f1-score']:.4f}")
    print(f"\n      Confusion Matrix:")
    print(f"      TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"      FN={cm[1][0]}  TP={cm[1][1]}")

    print("\n[4/4] Saving model artifacts to models/ ...")
    save_model(model, encoders, scaler, path='models/')
    print("      ✓ anomaly_model.pkl, encoders.pkl, scaler.pkl saved")

    print("\n" + "=" * 55)
    print("  DONE. Now run: python app.py")
    print("=" * 55 + "\n")


if __name__ == '__main__':
    main()
