from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor


def build_models(random_state: int):
    return {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=300, random_state=random_state),
        "gradient_boosting": GradientBoostingRegressor(random_state=random_state),
        "mlp": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=random_state),
    }
