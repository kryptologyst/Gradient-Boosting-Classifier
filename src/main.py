import typer
import sys
from loguru import logger

from .config import settings
from .data import load_classification_data
from .model import GradientBoostingModel
from .visualizer import GBVisualizer

app = typer.Typer(help="Gradient Boosting Classifier CLI")
logger.remove()
logger.add(sys.stderr, level=settings.log_level)


@app.command()
def train(
    dataset: str = typer.Option("wine", help="Dataset: iris, wine, breast_cancer"),
    n_estimators: int = typer.Option(100, help="Number of boosting stages"),
    lr: float = typer.Option(0.1, help="Learning rate"),
    max_depth: int = typer.Option(3, help="Max tree depth"),
    tune: bool = typer.Option(False, help="Run GridSearchCV"),
    visualize: bool = typer.Option(True, help="Generate plots"),
):
    logger.info(f"Training GB on {dataset} | trees={n_estimators} | lr={lr}")
    X_train, X_test, y_train, y_test, fn, cn = load_classification_data(dataset)
    model = GradientBoostingModel(n_estimators=n_estimators, learning_rate=lr, max_depth=max_depth)
    model.fit(X_train, y_train, fn, cn)
    results = model.evaluate(X_test, y_test)
    logger.info(f"Accuracy: {results['accuracy']:.2%}")
    logger.info(f"Feature Importance: {model.get_feature_importance()}")
    if tune:
        best = model.grid_search(X_train, y_train)
        logger.info(f"Best: {best['best_params']} → {best['best_score']:.3f}")
    if visualize:
        vis = GBVisualizer()
        vis.plot_feature_importance(model.get_feature_importance(), save_path=settings.plots_dir / "feature_importance.png")
        vis.plot_confusion_matrix(results["confusion_matrix"], cn, save_path=settings.plots_dir / "confusion_matrix.png")
    model.save(str(settings.models_dir / "gb_model.joblib"))
    logger.success("Training complete!")


if __name__ == "__main__":
    app()
