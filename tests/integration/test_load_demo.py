from ctgan import CTGAN, load_demo


def test_load_demo():
    """End-to-end test to load and synthesize data."""
    # Setup
    discrete_columns = ['Hop_count'
    ]
    ctgan = CTGAN(epochs=1000)

    # Run
    data = load_demo()
    ctgan.fit(data, discrete_columns)
    samples = ctgan.sample(1000)

    # Assert
    assert samples.shape == (1000, 15)
    assert all([col[0] != ' ' for col in samples.columns])
    assert not samples.isna().any().any()
