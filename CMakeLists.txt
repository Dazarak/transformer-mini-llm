cmake_minimum_required(VERSION 3.12)
project(TransformerFromScratch CXX)

# Utilisation du standard C++17 minimum
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Exécutable principal
add_executable(transformer src/main.cpp)

# Indiquer où trouver les fichiers d'en-tête (.h)
target_include_directories(transformer PRIVATE include)