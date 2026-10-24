all: run

compile:
	mkdir -p build resultats/sobel resultats/laplace resultats/canny
	cd build && cmake .. && $(MAKE)

run: compile
	./build/main_prog

clean:
	rm -rf build
