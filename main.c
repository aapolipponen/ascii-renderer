#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include <float.h>
#include <stdlib.h>
#include <stdbool.h>
#include <termios.h>

#define SCREEN_WIDTH 64
#define SCREEN_HEIGHT 32

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define TWO_PI (2.0 * M_PI)

#define CONTROL_LINE_HEIGHT 1

typedef enum {
    SPEED_ADJUSTMENT,
    DIRECT_ROTATION
} CameraControlMode;

CameraControlMode rotationMode = SPEED_ADJUSTMENT;

typedef struct {
    float yaw;
    float pitch;
    float roll;
    float deltaYaw;
    float deltaPitch;
    float deltaRoll;
} Rotation;

Rotation rotation = {
    .yaw = 0.0f,
    .pitch = 0.0f,
    .roll = 0.0f,
    .deltaYaw = 0.05f,
    .deltaPitch = 0.05f,
    .deltaRoll = 0.05f
};

float zoomSpeed = 1.0f;
float angleChange = 0.05f;

typedef struct {
    float centerX, centerY, centerZ;
    float focalLength;
} ScreenData;

typedef struct {
    float x, y, z;
} Point3D;

typedef Point3D Vertex3D;
typedef Point3D Vector3D;

typedef struct {
    Vertex3D ambient;   // Ambient reflection coefficients (RGB)
    Vertex3D diffuse;   // Diffuse reflection coefficients (RGB)
    Vertex3D specular;  // Specular reflection coefficients (RGB)
    float shininess;    // Shininess exponent
} Material;

Material material = {
    .ambient = (Vertex3D) {.x = 0.1f, .y = 0.1f, .z = 0.1f},
    .diffuse = (Vertex3D) {.x = 0.6f, .y = 0.6f, .z = 0.6f},
    .specular = (Vertex3D) {.x = 0.9f, .y = 0.9f, .z = 0.9f},
    .shininess = 32.0f
};

typedef struct {
    Vertex3D direction;   // Light direction
    Vertex3D ambient;     // Ambient light intensity
    Vertex3D diffuse;     // Diffuse light intensity
    Vertex3D specular;    // Specular light intensity
} Light;

Light light = {
    .direction = (Vertex3D) {.x = 1.0f, .y = 1.0f, .z = -1.0f},
    .ambient = (Vertex3D) {.x = 0.2f, .y = 0.2f, .z = 0.2f},
    .diffuse = (Vertex3D) {.x = 0.7f, .y = 0.7f, .z = 0.7f},
    .specular = (Vertex3D) {.x = 1.0f, .y = 1.0f, .z = 1.0f}
};

typedef struct {
    Vector3D position;
    float yaw;
    float pitch;
    float radius;
} Camera;

Vertex3D addVertices(Vertex3D v1, Vertex3D v2) {
    Vertex3D result;
    result.x = v1.x + v2.x;
    result.y = v1.y + v2.y;
    result.z = v1.z + v2.z;
    return result;
}

Vertex3D multiplyVertices(Vertex3D v1, Vertex3D v2) {
    Vertex3D result;
    result.x = v1.x * v2.x;
    result.y = v1.y * v2.y;
    result.z = v1.z * v2.z;
    return result;
}

Vertex3D scalarMultiplyVertex(Vertex3D v, float scalar) {
    Vertex3D result;
    result.x = v.x * scalar;
    result.y = v.y * scalar;
    result.z = v.z * scalar;
    return result;
}

Camera camera;

char screen[SCREEN_HEIGHT][SCREEN_WIDTH];
float zbuffer[SCREEN_HEIGHT][SCREEN_WIDTH];

float focalLength = 75.0f;
float centerX, centerY, centerZ;

char shades[] = ".:-=+*%@#";  // Characters for shading from darkest to brightest

typedef struct {
    Vertex3D* vertices;
    int* faces;
    int vertexCount;
    int faceCount;
    Vector3D minBounds;
    Vector3D maxBounds;
} Model;

void handleCameraControl(Camera* camera);
float computeFaceBrightness(Vertex3D vertex, Vertex3D normal, Light light, Vertex3D viewDirection, Material material);

void initZBuffer() {
    for (int y = 0; y < SCREEN_HEIGHT; y++) {
        for (int x = 0; x < SCREEN_WIDTH; x++) {
            zbuffer[y][x] = FLT_MAX;
        }
    }
}

void initScreen() {
    for (int y = 0; y < SCREEN_HEIGHT; y++) {
        for (int x = 0; x < SCREEN_WIDTH; x++) {
            screen[y][x] = ' ';
        }
    }
}

Model readObjFile(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        // Handle error
        exit(1);
    }

    Model model;
    model.vertexCount = 0;
    model.faceCount = 0;

    // Initial values for min/max bounds
    model.minBounds.x = model.minBounds.y = model.minBounds.z = FLT_MAX;
    model.maxBounds.x = model.maxBounds.y = model.maxBounds.z = -FLT_MAX;

    // First pass: count the number of vertices and faces
    char line[256];
    while (fgets(line, sizeof(line), file)) {
        if (line[0] == 'v') model.vertexCount++;
        if (line[0] == 'f') model.faceCount++;
    }

    model.vertices = malloc(model.vertexCount * sizeof(Vertex3D));
    model.faces = malloc(model.faceCount * 3 * sizeof(int));  // Assuming triangles

    fseek(file, 0, SEEK_SET);  // Go back to the start of the file

    // Initializing max/min values for x, y, z
    float maxX = -FLT_MAX, minX = FLT_MAX;
    float maxY = -FLT_MAX, minY = FLT_MAX;
    float maxZ = -FLT_MAX, minZ = FLT_MAX;

    // Second pass: read the vertices and faces
    int vIndex = 0, fIndex = 0;
    while (fgets(line, sizeof(line), file)) {
        if (line[0] == 'v') {
            sscanf(line, "v %f %f %f", &model.vertices[vIndex].x, &model.vertices[vIndex].y, &model.vertices[vIndex].z);

            // Check for max/min x, y, z values
            if (model.vertices[vIndex].x > model.maxBounds.x) model.maxBounds.x = model.vertices[vIndex].x;
            if (model.vertices[vIndex].x < model.minBounds.x) model.minBounds.x = model.vertices[vIndex].x;
            if (model.vertices[vIndex].y > model.maxBounds.y) model.maxBounds.y = model.vertices[vIndex].y;
            if (model.vertices[vIndex].y < model.minBounds.y) model.minBounds.y = model.vertices[vIndex].y;
            if (model.vertices[vIndex].z > model.maxBounds.z) model.maxBounds.z = model.vertices[vIndex].z;
            if (model.vertices[vIndex].z < model.minBounds.z) model.minBounds.z = model.vertices[vIndex].z;

            vIndex++;
        }        if (line[0] == 'f') {
            int v1, v2, v3;
            sscanf(line, "f %d %d %d", &v1, &v2, &v3);
            model.faces[fIndex * 3] = v1 - 1;  // .obj indices start at 1
            model.faces[fIndex * 3 + 1] = v2 - 1;
            model.faces[fIndex * 3 + 2] = v3 - 1;
            fIndex++;
        }
    }

    // Compute the center coordinates for x, y, z
    centerX = (maxX + minX) / 2;
    centerY = (maxY + minY) / 2;
    centerZ = (maxZ + minZ) / 2;

    fclose(file);
    return model;
}

typedef struct {
    int x, y;
} Vertex2D;

typedef struct {
    float m[4][4];
} Matrix4x4;

Vertex3D CrossProduct(Vertex3D a, Vertex3D b) {
    Vertex3D result;
    result.x = a.y * b.z - a.z * b.y;
    result.y = a.z * b.x - a.x * b.z;
    result.z = a.x * b.y - a.y * b.x;
    return result;
}

Vertex3D normalizeVertex(Vertex3D v) {
    float length = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    return (Vertex3D) {v.x / length, v.y / length, v.z / length};
}

float DotProduct(Vertex3D a, Vertex3D b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Matrix4x4 RotationMatrix(float ax, float ay, float az) {
    float sinX = sin(ax), cosX = cos(ax);
    float sinY = sin(ay), cosY = cos(ay);
    float sinZ = sin(az), cosZ = cos(az);

    Matrix4x4 matrix = {
        cosY * cosZ, 
        cosX * sinZ + sinX * sinY * cosZ,
        sinX * sinZ - cosX * sinY * cosZ,
        0,
        
        -cosY * sinZ,
        cosX * cosZ - sinX * sinY * sinZ,
        sinX * cosZ + cosX * sinY * sinZ,
        0,
        
        sinY,
        -sinX * cosY,
        cosX * cosY,
        0,
        
        0, 0, 0, 1
    };
    return matrix;
}

Matrix4x4 ScalingMatrix(float sx, float sy, float sz) {
    Matrix4x4 result = {0};  // Initializes all values to zero
    result.m[0][0] = sx;
    result.m[1][1] = sy;
    result.m[2][2] = sz;
    result.m[3][3] = 1;
    return result;
}

Matrix4x4 MultiplyMatrices(const Matrix4x4 a, const Matrix4x4 b) {
    Matrix4x4 result = {0};

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 4; k++) {
                result.m[i][j] += a.m[i][k] * b.m[k][j];
            }
        }
    }
    
    return result;
}

Matrix4x4 TranslationMatrix(float tx, float ty, float tz) {
    Matrix4x4 result = {{
        {1, 0, 0, tx},
        {0, 1, 0, ty},
        {0, 0, 1, tz},
        {0, 0, 0, 1}
    }};
    return result;
}


Vertex3D TransformVertex(Vertex3D vertex, Matrix4x4 matrix) {
    Vertex3D result;
    result.x = vertex.x * matrix.m[0][0] + vertex.y * matrix.m[1][0] + vertex.z * matrix.m[2][0] + matrix.m[3][0];
    result.y = vertex.x * matrix.m[0][1] + vertex.y * matrix.m[1][1] + vertex.z * matrix.m[2][1] + matrix.m[3][1];
    result.z = vertex.x * matrix.m[0][2] + vertex.y * matrix.m[1][2] + vertex.z * matrix.m[2][2] + matrix.m[3][2];
    return result;
}

Vertex2D ProjectVertexTo2D(Vertex3D v, Camera camera, float focalLength) {
    // Adjust the vertex by the camera position
    v.x -= camera.position.x;
    v.y -= camera.position.y;
    v.z -= camera.position.z;

    Vertex2D result;
    float offsetZ = v.z;
    
    static const float aspectRatio = (float)SCREEN_WIDTH / (float)SCREEN_HEIGHT;
    float projectedX = (v.x / offsetZ) * focalLength * aspectRatio;
    float projectedY = (v.y / offsetZ) * focalLength;

    result.x = projectedX + SCREEN_WIDTH / 2;
    result.y = -projectedY + SCREEN_HEIGHT / 2;

    return result;
}

void drawShadedLine(int x1, int y1, float z1, int x2, int y2, float z2, float brightness) {
    int shadeIndex = (int)((brightness * (sizeof(shades) - 2)) + 0.5);  
    char shade = shades[shadeIndex];
    
    int dx = abs(x2 - x1);
    int dy = abs(y2 - y1);
    int dz = z2 - z1;
    int sx = x1 < x2 ? 1 : -1;
    int sy = y1 < y2 ? 1 : -1;
    int sz = z1 < z2 ? 1 : -1;
    int err = (dx > dy ? dx : -dy) / 2;
    int e2;

    float zCurrent = z1;

    while (1) {
        if (x1 >= 0 && x1 < SCREEN_WIDTH && y1 >= 0 && y1 < SCREEN_HEIGHT) {
            if (zCurrent < zbuffer[y1][x1]) {  // Check against the Z-buffer
                screen[y1][x1] = shade;
                zbuffer[y1][x1] = zCurrent;
            }
        }

        if (x1 == x2 && y1 == y2) break;

        e2 = err;
        if (e2 > -dx) { err -= dy; x1 += sx; zCurrent += sz; }
        if (e2 < dy) { err += dx; y1 += sy; zCurrent += sz; }
    }
}

Matrix4x4 computeTransform(Rotation rotationData) {
    Matrix4x4 translationToOrigin = TranslationMatrix(-centerX, -centerY, -centerZ);
    Matrix4x4 rotationYaw = RotationMatrix(0, rotationData.yaw, 0);
    Matrix4x4 rotationPitch = RotationMatrix(rotationData.pitch, 0, 0);
    Matrix4x4 rotationRoll = RotationMatrix(0, 0, rotationData.roll);
    Matrix4x4 rotation = MultiplyMatrices(rotationYaw, MultiplyMatrices(rotationPitch, rotationRoll));
    Matrix4x4 translationBack = TranslationMatrix(centerX, centerY, centerZ);

    Matrix4x4 transform = MultiplyMatrices(translationBack, rotation);
    transform = MultiplyMatrices(transform, translationToOrigin);

    return transform;
}

float computeFaceBrightness(Vertex3D vertex, Vertex3D normal, Light light, Vertex3D viewDirection, Material material) {
    // Calculate the light direction
    Vertex3D lightDirection = normalizeVertex(light.direction);

    // Calculate dot product for diffuse component using Lambert's cosine law
    float dotNormalLight = DotProduct(normal, lightDirection);
    float diffuse = fmax(0.0, dotNormalLight);

    // Calculate the halfway vector (Blinn-Phong halfway vector) and dot product for specular component
    Vertex3D halfway = normalizeVertex(addVertices(lightDirection, viewDirection));
    float dotNormalHalfway = DotProduct(normal, halfway);
    float specular = pow(fmax(0.0, dotNormalHalfway), material.shininess);

    // Assuming ambient reflection as a constant scalar for simplicity
    float ambient = 0.1;

    // Calculate brightness as the weighted sum of components
    float brightness = ambient + diffuse + specular;

    // Ensure brightness is in the [0, 1] range
    brightness = fmax(0.0, fmin(1.0, brightness));

    return brightness;
}

void drawFace(Model model, Matrix4x4 transform, int faceIndex, float brightness, float focalLength) {
    for (int j = 0; j < 3; j++) {
        Vertex3D currentVertex = TransformVertex(model.vertices[model.faces[faceIndex * 3 + j]], transform);
        Vertex3D nextVertex = TransformVertex(model.vertices[model.faces[faceIndex * 3 + ((j + 1) % 3)]], transform);

        Vertex2D projectedCurrent = ProjectVertexTo2D(currentVertex, camera, focalLength);
        Vertex2D projectedNext = ProjectVertexTo2D(nextVertex, camera, focalLength);

        drawShadedLine(projectedCurrent.x, projectedCurrent.y, currentVertex.z, projectedNext.x, projectedNext.y, nextVertex.z, brightness);
    }
}

void printControlLine() {
    if (rotationMode == SPEED_ADJUSTMENT) {
        printf("\n[Speed Adjustment Mode]");
        printf("\nControls: w/s - Zoom | a/d - Adjust Yaw Speed | r/f - Adjust Pitch Speed | q/e - Adjust Roll Speed | Mode Change: Spacebar/M | Speeds: Yaw: %.2f, Pitch: %.2f, Roll: %.2f\n",
               rotation.deltaYaw, rotation.deltaPitch, rotation.deltaRoll);
    } else if (rotationMode == DIRECT_ROTATION) {
        printf("\n[Direct Rotation Mode]");
        printf("\nControls: w/s - Zoom | a/d - Rotate Yaw | r/f - Rotate Pitch | q/e - Rotate Roll | Mode Change: Spacebar/M\n");
    }
}

void printScreen() {
    char buffer[(SCREEN_HEIGHT + CONTROL_LINE_HEIGHT) * (SCREEN_WIDTH + 1) + 1];  // +1 for each newline and +1 for the null terminator
    char* p = buffer;

    for (int y = 0; y < SCREEN_HEIGHT; y++) {
        for (int x = 0; x < SCREEN_WIDTH; x++) {
            *p++ = screen[y][x] ? screen[y][x] : ' ';
        }
        *p++ = '\n';
    }
    *p = '\0';
    system("clear");
    printf("%s", buffer);
    printControlLine();
}

void clearScreen() {
    for (int y = 0; y < SCREEN_HEIGHT; y++) {
        for (int x = 0; x < SCREEN_WIDTH; x++) {
            screen[y][x] = ' ';
        }
    }
}

void renderLoop(Model model) {
    Vertex3D lightDirection = {0, 0, -1}; 
    lightDirection = normalizeVertex(lightDirection);

    int sleeptime = 50000;

    float depth = model.maxBounds.z - model.minBounds.z;
    float additionalDistance = depth * -5;
    camera.position.z = model.maxBounds.z + depth / 2 + additionalDistance;

    while (1) {
        handleCameraControl(&camera);
        clearScreen();
        initZBuffer();
        Matrix4x4 transform = computeTransform(rotation);
        // For each face
        for (int i = 0; i < model.faceCount; i++) {
            // Calculate face normal
            Vertex3D vertexA = model.vertices[model.faces[i * 3]];
            Vertex3D vertexB = model.vertices[model.faces[i * 3 + 1]];
            Vertex3D vertexC = model.vertices[model.faces[i * 3 + 2]];

            Vertex3D side1 = {vertexB.x - vertexA.x, vertexB.y - vertexA.y, vertexB.z - vertexA.z};
            Vertex3D side2 = {vertexC.x - vertexA.x, vertexC.y - vertexA.y, vertexC.z - vertexA.z};

            Vertex3D normal = CrossProduct(side1, side2);
            normal = normalizeVertex(normal);

            // Calculate brightness for this face
            float brightness = computeFaceBrightness(vertexA, normal, light, normal, material);

            // Draw the face with the calculated brightness
            drawFace(model, transform, i, brightness, focalLength);
        }
        rotation.yaw += rotation.deltaYaw;
        if (rotation.yaw >= TWO_PI) {
            rotation.yaw -= TWO_PI;
        }
        rotation.pitch += rotation.deltaPitch;
        if (rotation.pitch >= TWO_PI) {
            rotation.pitch -= TWO_PI;
        }
        rotation.roll += rotation.deltaRoll;
        if (rotation.roll >= TWO_PI) {
            rotation.roll -= TWO_PI;
        }

        printScreen();
        usleep(sleeptime);
    }
}

void setupTerminal(struct termios* oldSettings) {
    struct termios newSettings;
    tcgetattr(fileno(stdin), oldSettings);
    newSettings = *oldSettings;
    newSettings.c_lflag &= (~ICANON & ~ECHO);
    tcsetattr(fileno(stdin), TCSANOW, &newSettings);
}

void handleCameraControl(Camera* camera) {
    struct termios oldSettings;
    setupTerminal(&oldSettings);

    fd_set set;
    struct timeval timeout;
    timeout.tv_sec = 0;
    timeout.tv_usec = 10000;  // Check for input every 10ms

    FD_ZERO(&set);
    FD_SET(fileno(stdin), &set);

    char c = 0;

    // Check for input
    if (select(FD_SETSIZE, &set, NULL, NULL, &timeout) > 0) {
        read(fileno(stdin), &c, 1);
    }

    if (c == ' ' || c == 'm') {  // Toggle the mode using spacebar or 'm'
        rotationMode = (rotationMode == SPEED_ADJUSTMENT) ? DIRECT_ROTATION : SPEED_ADJUSTMENT;
        
        if (rotationMode == DIRECT_ROTATION) {
            // Zero out the speeds when switching to DIRECT_ROTATION mode
            rotation.deltaYaw = 0;
            rotation.deltaPitch = 0;
            rotation.deltaRoll = 0;
        }
    } else {
        // Move zoom controls out of the switch statement to allow zooming in both modes
        switch (c) {
            case 'w':
                camera->position.z -= zoomSpeed;
                break;
            case 's':
                camera->position.z += zoomSpeed;
                break;
        }

        switch (rotationMode) {
            case SPEED_ADJUSTMENT:
                switch (c) {
                    case 'a':
                        rotation.deltaYaw -= angleChange;  // Adjust yaw speed left
                        break;
                    case 'd':
                        rotation.deltaYaw += angleChange;  // Adjust yaw speed right
                        break;
                    case 'q':
                        rotation.deltaRoll -= angleChange;  // Adjust roll speed left
                        break;
                    case 'e':
                        rotation.deltaRoll += angleChange;  // Adjust roll speed right
                        break;
                    case 'r':
                        rotation.deltaPitch += angleChange;  // Adjust pitch speed up
                        break;
                    case 'f':
                        rotation.deltaPitch -= angleChange;  // Adjust pitch speed down
                        break;
                }
                break;

            case DIRECT_ROTATION:
                switch (c) {
                    case 'a':
                        rotation.yaw -= angleChange;  // Rotate camera left directly
                        break;
                    case 'd':
                        rotation.yaw += angleChange;  // Rotate camera right directly
                        break;
                    case 'q':
                        rotation.roll -= angleChange;  // Rotate model roll-wise left directly
                        break;
                    case 'e':
                        rotation.roll += angleChange;  // Rotate model roll-wise right directly
                        break;
                    case 'r':
                        rotation.pitch += angleChange;  // Rotate model pitch-wise up directly
                        break;
                    case 'f':
                        rotation.pitch -= angleChange;  // Rotate model pitch-wise down directly
                        break;
                }
                break;
        }
    }

    // Reset terminal settings
    tcsetattr(fileno(stdin), TCSANOW, &oldSettings);
}

int main() {
    Model model = readObjFile("models/utahteapot.obj");
    renderLoop(model);
    return 0;
}