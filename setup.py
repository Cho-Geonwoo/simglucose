from setuptools import setup

setup(
    name="simglucose",
    version="0.3.3",
    description="A Type-1 Diabetes Simulator as a Reinforcement Learning Environment in OpenAI gym or rllab (python implementation of UVa/Padova Simulator)",
    url="https://github.com/Cho-Geonwoo/simglucose",
    author="Geonwoo Cho",
    author_email="geonwoo@umich.edu",
    license="MIT",
    packages=["simglucose"],
    install_requires=["pandas", "numpy", "scipy", "matplotlib", "pathos", "gym==0.9.4"],
    include_package_data=True,
    zip_safe=False,
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
