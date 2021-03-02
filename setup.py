import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="DANet",
    version="0.0.4",
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