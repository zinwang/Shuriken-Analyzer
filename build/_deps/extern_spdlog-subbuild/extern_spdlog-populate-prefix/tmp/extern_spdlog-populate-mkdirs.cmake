# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/jensen/lab/Shuriken-Analyzer/build/_deps/extern_spdlog-src")
  file(MAKE_DIRECTORY "/home/jensen/lab/Shuriken-Analyzer/build/_deps/extern_spdlog-src")
endif()
file(MAKE_DIRECTORY
  "/home/jensen/lab/Shuriken-Analyzer/build/_deps/extern_spdlog-build"
  "/home/jensen/lab/Shuriken-Analyzer/build/_deps/extern_spdlog-subbuild/extern_spdlog-populate-prefix"
  "/home/jensen/lab/Shuriken-Analyzer/build/_deps/extern_spdlog-subbuild/extern_spdlog-populate-prefix/tmp"
  "/home/jensen/lab/Shuriken-Analyzer/build/_deps/extern_spdlog-subbuild/extern_spdlog-populate-prefix/src/extern_spdlog-populate-stamp"
  "/home/jensen/lab/Shuriken-Analyzer/build/_deps/extern_spdlog-subbuild/extern_spdlog-populate-prefix/src"
  "/home/jensen/lab/Shuriken-Analyzer/build/_deps/extern_spdlog-subbuild/extern_spdlog-populate-prefix/src/extern_spdlog-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/jensen/lab/Shuriken-Analyzer/build/_deps/extern_spdlog-subbuild/extern_spdlog-populate-prefix/src/extern_spdlog-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/jensen/lab/Shuriken-Analyzer/build/_deps/extern_spdlog-subbuild/extern_spdlog-populate-prefix/src/extern_spdlog-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
