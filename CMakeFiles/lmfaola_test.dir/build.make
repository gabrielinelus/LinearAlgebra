# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/gabrielinelus/Documents/MastersProject/LMFAO/LinearAlgebra

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/gabrielinelus/Documents/MastersProject/LMFAO/LinearAlgebra

# Include any dependencies generated for this target.
include CMakeFiles/lmfaola_test.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/lmfaola_test.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/lmfaola_test.dir/flags.make

CMakeFiles/lmfaola_test.dir/tests/test.cpp.o: CMakeFiles/lmfaola_test.dir/flags.make
CMakeFiles/lmfaola_test.dir/tests/test.cpp.o: tests/test.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/gabrielinelus/Documents/MastersProject/LMFAO/LinearAlgebra/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/lmfaola_test.dir/tests/test.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/lmfaola_test.dir/tests/test.cpp.o -c /home/gabrielinelus/Documents/MastersProject/LMFAO/LinearAlgebra/tests/test.cpp

CMakeFiles/lmfaola_test.dir/tests/test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/lmfaola_test.dir/tests/test.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/gabrielinelus/Documents/MastersProject/LMFAO/LinearAlgebra/tests/test.cpp > CMakeFiles/lmfaola_test.dir/tests/test.cpp.i

CMakeFiles/lmfaola_test.dir/tests/test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/lmfaola_test.dir/tests/test.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/gabrielinelus/Documents/MastersProject/LMFAO/LinearAlgebra/tests/test.cpp -o CMakeFiles/lmfaola_test.dir/tests/test.cpp.s

CMakeFiles/lmfaola_test.dir/tests/test.cpp.o.requires:

.PHONY : CMakeFiles/lmfaola_test.dir/tests/test.cpp.o.requires

CMakeFiles/lmfaola_test.dir/tests/test.cpp.o.provides: CMakeFiles/lmfaola_test.dir/tests/test.cpp.o.requires
	$(MAKE) -f CMakeFiles/lmfaola_test.dir/build.make CMakeFiles/lmfaola_test.dir/tests/test.cpp.o.provides.build
.PHONY : CMakeFiles/lmfaola_test.dir/tests/test.cpp.o.provides

CMakeFiles/lmfaola_test.dir/tests/test.cpp.o.provides.build: CMakeFiles/lmfaola_test.dir/tests/test.cpp.o


# Object files for target lmfaola_test
lmfaola_test_OBJECTS = \
"CMakeFiles/lmfaola_test.dir/tests/test.cpp.o"

# External object files for target lmfaola_test
lmfaola_test_EXTERNAL_OBJECTS =

lmfaola_test: CMakeFiles/lmfaola_test.dir/tests/test.cpp.o
lmfaola_test: CMakeFiles/lmfaola_test.dir/build.make
lmfaola_test: /usr/local/lib/libboost_system.so
lmfaola_test: /usr/local/lib/libboost_iostreams.so
lmfaola_test: /usr/local/lib/libboost_program_options.so
lmfaola_test: /usr/local/lib/libboost_regex.so
lmfaola_test: /usr/lib/libgtest.a
lmfaola_test: CMakeFiles/lmfaola_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/gabrielinelus/Documents/MastersProject/LMFAO/LinearAlgebra/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable lmfaola_test"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/lmfaola_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/lmfaola_test.dir/build: lmfaola_test

.PHONY : CMakeFiles/lmfaola_test.dir/build

CMakeFiles/lmfaola_test.dir/requires: CMakeFiles/lmfaola_test.dir/tests/test.cpp.o.requires

.PHONY : CMakeFiles/lmfaola_test.dir/requires

CMakeFiles/lmfaola_test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/lmfaola_test.dir/cmake_clean.cmake
.PHONY : CMakeFiles/lmfaola_test.dir/clean

CMakeFiles/lmfaola_test.dir/depend:
	cd /home/gabrielinelus/Documents/MastersProject/LMFAO/LinearAlgebra && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/gabrielinelus/Documents/MastersProject/LMFAO/LinearAlgebra /home/gabrielinelus/Documents/MastersProject/LMFAO/LinearAlgebra /home/gabrielinelus/Documents/MastersProject/LMFAO/LinearAlgebra /home/gabrielinelus/Documents/MastersProject/LMFAO/LinearAlgebra /home/gabrielinelus/Documents/MastersProject/LMFAO/LinearAlgebra/CMakeFiles/lmfaola_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/lmfaola_test.dir/depend
