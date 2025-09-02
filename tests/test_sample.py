def test_basic_math():
    assert 2 + 2 == 4

def test_dummy_model():
    # Dummy test untuk pastikan pipeline jalan
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression

    X, y = load_iris(return_X_y=True)
    clf = LogisticRegression(max_iter=200)
    clf.fit(X, y)

    assert clf.score(X, y) > 0.8
