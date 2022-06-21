import setuptools

NAME = "pandisc"
VERSION = "1.1"

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

if __name__ == "__main__":
    setuptools.setup(
        name=NAME,
        version=VERSION,
        author="Bo Peng",
        author_email="bp392@cornell.edu",
        description="PANDISC model for the integrated H I line.",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/bpqdbpqd/pandisc",
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
            package_dir={"": "src"},
            packages=setuptools.find_packages(where="src"),
            python_requires=">=3.6",
            install_requires=[
                "numpy>=1.13",
            ]
    )
