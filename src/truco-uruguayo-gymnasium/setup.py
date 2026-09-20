from setuptools import setup

setup(
    name="truco-uruguayo-gymnasium",
    version="0.0.1",
    install_requires=["gymnasium==0.29.1", "numpy==2.0.1"],
    entry_points={
        "console_scripts": [
            "truco_uruguayo_gymnasium = truco_uruguayo_gymnasium.cli:main",
        ],
    },
)
