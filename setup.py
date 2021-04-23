import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="DANet",
    version="0.0.5",
    install_requires=["tensorflow>=2.4.1","museval>=0.4.0","soundfile>=0.10.3.post1",
                      "pandas>=1.1.5","numpy>=1.19.5","scipy>=1.4.1","librosa>=0.8.0",
                      "matplotlib>=3.2.2"],
    author="KMASAHIRO",
    description="a model that separates a mixed sound into whatever kinds of sound you like",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KMASAHIRO/DANet",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7.10',
)