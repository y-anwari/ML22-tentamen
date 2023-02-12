from pathlib import Path
from typing import Union

from pydantic import BaseModel, HttpUrl
from ray import tune

SAMPLE_INT = tune.search.sample.Integer
SAMPLE_FLOAT = tune.search.sample.Float

cwd = Path(__file__)
root = (cwd / "../..").resolve()


class Settings(BaseModel):
    datadir: Path
    testurl: HttpUrl
    trainurl: HttpUrl
    testfile: Path
    trainfile: Path
    modeldir: Path
    logdir: Path
    modelname: str
    batchsize: int


presets = Settings(
    datadir=root / "data/raw",
    testurl="https://archive.ics.uci.edu/ml/machine-learning-databases/00195/Test_Arabic_Digit.txt",  # noqa N501
    trainurl="https://archive.ics.uci.edu/ml/machine-learning-databases/00195/Train_Arabic_Digit.txt",  # noqa N501
    testfile=Path("ArabicTest.txt"),
    trainfile=Path("ArabicTrain.txt"),
    modeldir=root / "models",
    logdir=root / "logs",
    modelname="model.pt",
    batchsize=128,
)


class BaseSearchSpace(BaseModel):
    input: int
    output: int
    tunedir: Path

    class Config:
        arbitrary_types_allowed = True


class LinearConfig(BaseSearchSpace):
    h1: int
    h2: int
    h3: int
    dropout: float


class LinearSearchSpace(BaseSearchSpace):
    h1: Union[int, SAMPLE_INT] = tune.randint(16, 128) #32 en 64 filters aan toegevoegd
    h2: Union[int, SAMPLE_INT] = tune.randint(16, 128) #32 en 64 filters aan toegevoegd
    h3: Union[int, SAMPLE_INT] = tune.randint(16, 128) #32 en 64 filters aan toegevoegd
    dropout: Union[float, SAMPLE_FLOAT] = tune.uniform(0.0, 0.3) # dropout van .5 naar .1 gemaakt.
