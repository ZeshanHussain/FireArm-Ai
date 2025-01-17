#include <FL/Fl.H>
#include <FL/Fl_Window.H>
#include "darknet.hpp"
#include "darknet_cfg_and_state.hpp"
#include <FL/Fl_Box.H>
#include <FL/Fl_Button.H>
#include <FL/Fl_Toggle_Button.H>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <chrono>
#include <iomanip>

class WebcamWindow : public Fl_Window {
private:
    Fl_Box* videoBox;
    cv::VideoCapture cap;
    cv::Mat frame, resizedFrame;
    Fl_RGB_Image* img;
    bool stopFlag;
    int argc;
    char** argv;

    // Neural network
    Darknet::NetworkPtr net;

    // FPS and statistics
    double estimated_fps;
    size_t frame_counter;
    size_t total_objects_found;
    std::chrono::high_resolution_clock::time_point timestamp_start;

    // Top bar widgets
    Fl_Box* topBar;
    Fl_Button* exitButton;

public:
    WebcamWindow(int w, int h, const char* title, int argc, char** argv)
        : Fl_Window(w, h, title), cap(0), img(nullptr), stopFlag(false), argc(argc), argv(argv),
          frame_counter(0), total_objects_found(0) {
        // Initialize the neural network
        Darknet::Parms parms = Darknet::parse_arguments(argc, argv);
        net = Darknet::load_neural_network(parms);

        // Open and configure the camera
        cap.open(0, cv::CAP_V4L2); // Use V4L2 backend for faster capture on Linux
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);  // Lower resolution for faster processing
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        cap.set(cv::CAP_PROP_FPS, 30);  // Set desired frame rate

        // Estimate FPS
        estimated_fps = estimate_camera_fps(cap);
        timestamp_start = std::chrono::high_resolution_clock::now();

        // Set a darker color scheme
        Fl::scheme("gtk+");

        // Set the window color
        color(FL_DARK_YELLOW);

        int frameWidth = w * 0.85;
        int buttonWidth = w * 0.2;
        int margin = 30;
        int topBarHeight = 30;

        // Create the top bar
        topBar = new Fl_Box(0, 0, w, topBarHeight, "GuardianSafe");
        topBar->box(FL_DOWN_BOX);
        topBar->color(FL_DARK_MAGENTA);

        // Create the exit button
        exitButton = new Fl_Button(w - 34, 2.5, 30, 25, "X");
        exitButton->box(FL_DOWN_BOX);
        exitButton->color(FL_RED);
        exitButton->callback([](Fl_Widget* widget, void* data) {
            ((WebcamWindow*)data)->hide();
        }, this);

        videoBox = new Fl_Box(margin, topBarHeight + margin, frameWidth - 2 * margin, h - topBarHeight - 2 * margin);
        videoBox->color(FL_BLACK);

        resizedFrame = cv::Mat(videoBox->h(), videoBox->w(), CV_8UC3);

        end();
    }

    double estimate_camera_fps(cv::VideoCapture & cap) {
        std::cout << "Estimating FPS..." << std::endl;

        // Read and discard a few frames to allow the camera to stabilize
        cv::Mat mat;
        for (int i = 0; i < 5; i++) {
            cap >> mat;
        }

        // Estimate FPS by reading several consecutive frames
        size_t frame_counter = 0;
        const auto ts1 = std::chrono::high_resolution_clock::now();
        for (int i = 0; cap.isOpened() and i < 5; i++) {
            cap >> mat;
            if (!mat.empty()) {
                frame_counter++;
            }
        }
        const auto ts2 = std::chrono::high_resolution_clock::now();

        const double actual_fps = static_cast<double>(frame_counter) / std::chrono::duration_cast<std::chrono::nanoseconds>(ts2 - ts1).count() * 1000000000.0;
        return actual_fps;
    }

    void drawAppleStyleText(cv::Mat& frame, const std::string& text, cv::Point position, double fontScale, int thickness) {
        // Calculate the size of the text
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline);

        // Define the background rectangle for the text
        cv::Rect bgRect(
            position.x, // X-coordinate of the top-left corner
            position.y - textSize.height, // Y-coordinate of the top-left corner
            textSize.width, // Width of the rectangle
            textSize.height + baseline // Height of the rectangle
        );

        // Create a semi-transparent black background for the text
        cv::Mat overlay = frame.clone();
        cv::rectangle(overlay, bgRect, cv::Scalar(0, 0, 0), cv::FILLED); // Black rectangle
        cv::addWeighted(overlay, 0.3, frame, 0.7, 0, frame); // Blend with the original frame (30% opacity)

        // Draw the text with a subtle shadow
        cv::putText(frame, text, position + cv::Point(2, 2), cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(0, 0, 0), thickness); // Shadow
        cv::putText(frame, text, position, cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(255, 120, 255), thickness); // Main text
    }

    void updateFrame() {
        if (!cap.read(frame)) {
            std::cerr << "Error: Failed to capture frame!" << std::endl;
            return;
        }

        // Process the frame through the neural network
        const auto results = Darknet::predict_and_annotate(net, frame);
        total_objects_found += results.size();

        // Calculate FPS
        const auto now = std::chrono::high_resolution_clock::now();
        const double elapsed_seconds = std::chrono::duration_cast<std::chrono::nanoseconds>(now - timestamp_start).count() / 1000000000.0;
        const double current_fps = frame_counter / elapsed_seconds;

        // Prepare multi-line text
        std::stringstream stats;
        stats << "FPS: " << std::fixed << std::setprecision(1) << current_fps << "\n"
              << "Objects: " << total_objects_found << "\n"
              << "Frame: " << frame_counter;

        // Split the text into lines
        std::vector<std::string> lines;
        std::string line;
        while (std::getline(stats, line, '\n')) {
            lines.push_back(line);
        }

        // Render each line with Apple-style text
        int y_offset = 30; // Starting Y position for the first line
        for (const auto& text_line : lines) {
            drawAppleStyleText(frame, text_line, cv::Point(10, y_offset), 0.7, 2);
            y_offset += 30; // Increment Y position for the next line
        }

        // Convert the frame to RGB for display
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
        cv::resize(frame, resizedFrame, resizedFrame.size());

        if (img) {
            delete img;
        }
        img = new Fl_RGB_Image(resizedFrame.data, resizedFrame.cols, resizedFrame.rows, 3);

        videoBox->image(img);
        videoBox->redraw();

        frame_counter++;
    }

    static void captureLoop(void* userdata) {
        WebcamWindow* win = static_cast<WebcamWindow*>(userdata);
        while (!win->stopFlag) {
            win->updateFrame();
            Fl::check();
            std::this_thread::sleep_for(std::chrono::milliseconds(1000 / 30)); // Aim for 30 FPS
        }
    }

    void startCapture() {
        std::thread captureThread(captureLoop, this);
        captureThread.detach();
    }

    ~WebcamWindow() {
        stopFlag = true;
        if (img) {
            delete img;
        }
        cap.release();
        Darknet::free_neural_network(net);
    }
};

int main(int argc, char* argv[]) {
    WebcamWindow window(640, 480, "FLTK Webcam", argc, argv);
    window.show();
    window.startCapture();
    return Fl::run();
}
