git submodule update --init --recursive

mkdir _build
cd _build
call cmake ..

cd ../blurhash
copy test2.png ..\_build

PAUSE