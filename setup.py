import os
import sys
from setuptools import setup, find_packages


with open('README.md') as readme_file:
    readme = readme_file.read()


setup_requirements = ['pytest-runner', ]


test_requirements = ['pytest>=3', ]


requirements = ['argh',]


COMMANDS = [
    'greet = chat.cli:greet',
]


if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist upload')
    sys.exit()


setup(
    author="Todd Young",
    author_email='youngmt1@ornl.gov',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Data parallelism study for natural language processing.",
    entry_points={'console_scripts': COMMANDS},
    install_requires=requirements,
    license="BSD license",
    long_description=readme,
    include_package_data=True,
    keywords='chat',
    name='le_chat',
    packages=find_packages(include=['chat', 'chat.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/yngtodd/chat',
    version='0.1.0',
    zip_safe=False,
)
