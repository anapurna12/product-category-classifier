import argparse
import joblib

def load_model(model_path: str):
    return joblib.load(model_path)

def predict_single(model, title: str) -> str:
    pred = model.predict([title])
    return pred[0]

def interactive_loop(model):
    print("=== Product Category Predictor ===")
    print("Unesi naziv proizvoda ili 'exit' za izlaz.")
    while True:
        try:
            user_input = input("\nNaslov proizvoda: ").strip()
        except EOFError:
            break

        if user_input.lower() in ["exit", "quit", "q"]:
            print("Zatvaram.")
            break

        if not user_input:
            print("Prazan unos, probaj ponovo.")
            continue

        category = predict_single(model, user_input)
        print(f"Predviđena kategorija: {category}")

def main(args):
    model = load_model(args.model_path)

    if args.title is not None:
        cat = predict_single(model, args.title)
        print(cat)
        return

    interactive_loop(model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Učitaj već istrenirani model i predvidi kategoriju za nove proizvode."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/product_category_model.pkl",
        help="Putanja do .pkl fajla modela"
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="(Opcionalno) Jedan naslov proizvoda za instant predikciju."
    )

    args = parser.parse_args()
    main(args)
