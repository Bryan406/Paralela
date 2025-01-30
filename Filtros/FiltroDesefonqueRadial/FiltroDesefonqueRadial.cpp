#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <windows.h>
#include <omp.h>
#include <chrono>

using namespace cv;
using namespace std;
using namespace chrono;

const string INPUT_FOLDER = "C:\\Users\\Bryan406\\Downloads\\Proyecto paralela 2P\\proyectoParalela\\ProyectoP2\\imagenesSinProcesar\\";
const string OUTPUT_FOLDER = "C:\\Users\\Bryan406\\Downloads\\Proyecto paralela 2P\\proyectoParalela\\ProyectoP2\\imagenesProcesadas\\";

vector<string> getFilesInDirectory(const string& folder) {
    vector<string> files;
    string searchPath = folder + "*.jpg";

    WIN32_FIND_DATAA fd;
    HANDLE hFind = FindFirstFileA(searchPath.c_str(), &fd);

    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            files.push_back(fd.cFileName);
        } while (FindNextFileA(hFind, &fd) != 0);
        FindClose(hFind);
    }
    else {
        cerr << "Error al abrir la carpeta: " << folder << endl;
    }
    return files;
}

// Filtro Sepia
void applySepia(Mat& img) {
    Mat kernel = (Mat_<float>(3, 3) <<
        0.272, 0.534, 0.131,
        0.349, 0.686, 0.168,
        0.393, 0.769, 0.189);
    transform(img, img, kernel);
}

// Filtro Desenfoque Gaussiano
void applyGaussianBlur(Mat& img) {
    GaussianBlur(img, img, Size(25, 25), 10);
}

// Filtro Canny (Detección de bordes)
void applyCanny(Mat& img) {
    Mat edges;
    Canny(img, edges, 100, 200);
    img = edges;
}

// Filtro de Rotación de 45 grados
void applyRotation(Mat& img) {
    // Obtener el centro de la imagen
    Point2f center(img.cols / 2.0F, img.rows / 2.0F);

    // Obtener la matriz de rotación para 45 grados
    Mat rotationMatrix = getRotationMatrix2D(center, 45.0, 1.0);

    // Aplicar la transformación de rotación
    warpAffine(img, img, rotationMatrix, img.size());
}

// Procesamiento en paralelo con OpenMP
void processImagesParallel(const vector<string>& inputFiles, int filterChoice) {
    int numThreads = omp_get_max_threads();
    cout << "Ejecutando en paralelo con " << numThreads << " hilos.\n";

#pragma omp parallel for schedule(guided)
    for (int i = 0; i < (int)inputFiles.size(); i++) {
        string inputPath = INPUT_FOLDER + inputFiles[i];
        string outputPath = OUTPUT_FOLDER + inputFiles[i];

        Mat img = imread(inputPath);
        if (img.empty()) {
            cerr << "Error al cargar la imagen: " << inputPath << endl;
            continue;
        }

        switch (filterChoice) {
        case 1: applySepia(img); break;
        case 2: applyGaussianBlur(img); break;
        case 3: applyRotation(img); break; // Rotación en lugar de mediana
        case 4: applyCanny(img); break;
        }

        imwrite(outputPath, img);

#pragma omp critical
        cout << "Imagen procesada en el hilo " << omp_get_thread_num() << ": " << inputFiles[i] << endl;
    }
}

int main() {
    int filterChoice, modeChoice;

    cout << "Selecciona el filtro que quieres aplicar:\n";
    cout << "1. Sepia\n";
    cout << "2. Desenfoque Gaussiano\n";
    cout << "3. Rotacion de 45 grados\n";
    cout << "4. Deteccion de Bordes Canny\n";
    cout << "Elige una opcion: ";
    cin >> filterChoice;

    if (filterChoice < 1 || filterChoice > 4) {
        cout << "Opcion de filtro no valida." << endl;
        return 0;
    }

    cout << "Selecciona el metodo de procesamiento:\n";
    cout << "1. Secuencial\n";
    cout << "2. Paralelo (OpenMP)\n";
    cout << "Elige una opcion: ";
    cin >> modeChoice;

    vector<string> files = getFilesInDirectory(INPUT_FOLDER);
    if (files.empty()) {
        cout << "No se encontraron imagenes en la carpeta de entrada." << endl;
        return 0;
    }

    auto start = high_resolution_clock::now();

    if (modeChoice == 1) {
        // Procesamiento Secuencial
        for (const string& file : files) {
            string inputPath = INPUT_FOLDER + file;
            string outputPath = OUTPUT_FOLDER + file;

            Mat img = imread(inputPath);
            if (img.empty()) {
                cerr << "Error al cargar la imagen: " << inputPath << endl;
                continue;
            }

            switch (filterChoice) {
            case 1: applySepia(img); break;
            case 2: applyGaussianBlur(img); break;
            case 3: applyRotation(img); break; // Rotación en lugar de mediana
            case 4: applyCanny(img); break;
            }

            imwrite(outputPath, img);
        }
    }
    else if (modeChoice == 2) {
        // Procesamiento en Paralelo
        processImagesParallel(files, filterChoice);
    }
    else {
        cout << "Opcion no valida." << endl;
        return 0;
    }

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);

    cout << "Filtro aplicado a todas las imagenes en la carpeta de salida." << endl;
    cout << "Tiempo total: " << duration.count() / 1000.0 << " segundos." << endl;

    return 0;
}
