from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="gpt2-text-generation",
    version="0.1.0",
    author="ayus1234",
    author_email="author@example.com",
    description="Text Generation with GPT-2 using various decoding methods",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ayus1234/Text_Generation_with_GPT_2",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "gpt2-demo=text_generation_demo:main",
            "gpt2-finetune=finetune_gpt2:main",
            "gpt2-generate=generate_from_finetuned:main",
        ],
    },
) 