#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

int main() {
    std::string videoPath = "sample3.mp4"; // Replace with your video path
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file.\n";
        return -1;
    }

    double fps = cap.get(cv::CAP_PROP_FPS);
    int frameCount = 0;

    double motionThreshold = 4.0;      // High-motion threshold
    double endMotionThreshold = 0.8;  // Low-motion threshold
    int highMotionPersistence = 5;    // Frames to confirm ad start
    int lowMotionPersistence = 15;    // Frames to confirm ad end
    int maxLowMotionGap = 10;         // Allowable low-motion gap within an ad
    double adFusionGap = 5.0;         // Maximum gap between two ads to merge them (in seconds)
    double preStartBuffer = 3.0;      // Buffer to look back for slower motion

    int highMotionCounter = 0;
    int lowMotionCounter = 0;
    int lowMotionGapCounter = 0;
    bool inAdSegment = false;

    std::vector<std::pair<double, double>> adSegments; // Store ad start and end timestamps

    std::cout << "Analyzing video for motion activity...\n";

    cv::Mat prevFrame, currFrame, flow;
    while (true) {
        cap >> currFrame;
        if (currFrame.empty()) break;

        frameCount++;
        cv::Mat currGray;
        cv::cvtColor(currFrame, currGray, cv::COLOR_BGR2GRAY);

        if (!prevFrame.empty()) {
            // Compute dense optical flow using Farneback method
            cv::calcOpticalFlowFarneback(prevFrame, currGray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

            // Calculate motion magnitude
            cv::Mat flowParts[2];
            cv::split(flow, flowParts);
            cv::Mat magnitude, angle;
            cv::cartToPolar(flowParts[0], flowParts[1], magnitude, angle, true);
            double motionMagnitude = cv::sum(magnitude)[0] / (currGray.rows * currGray.cols);

            // Get timestamp
            double timestamp = frameCount / fps;

            // Detect high motion (ad start)
            if (motionMagnitude > motionThreshold) {
                highMotionCounter++;
                lowMotionCounter = 0;
                lowMotionGapCounter = 0;

                if (highMotionCounter >= highMotionPersistence && !inAdSegment) {
                    // Apply pre-start buffer
                    double adStart = std::max(0.0, timestamp - preStartBuffer);
                    adSegments.push_back({adStart, 0});
                    inAdSegment = true;
                    std::cout << "Ad started at timestamp: " << adStart << " seconds.\n";
                }
            }
            // Detect low motion (ad end)
            else if (inAdSegment && motionMagnitude < endMotionThreshold) {
                lowMotionCounter++;
                highMotionCounter = 0;

                if (lowMotionCounter <= maxLowMotionGap) {
                    lowMotionGapCounter++;
                } else if (lowMotionCounter >= lowMotionPersistence) {
                    adSegments.back().second = timestamp;
                    std::cout << "Ad ended at timestamp: " << timestamp << " seconds.\n";
                    inAdSegment = false;
                    lowMotionCounter = 0;
                }
            }
            // Reset counters if motion is in-between thresholds
            else {
                highMotionCounter = 0;
                lowMotionCounter = 0;
                lowMotionGapCounter = 0;
            }
        }
        prevFrame = currGray.clone();
    }

    cap.release();

    // Merge adjacent ad segments if the gap is small
    for (size_t i = 1; i < adSegments.size(); ++i) {
        if (adSegments[i].first - adSegments[i - 1].second <= adFusionGap) {
            adSegments[i - 1].second = adSegments[i].second;
            adSegments.erase(adSegments.begin() + i);
            --i;
        }
    }

    std::cout << "Merged ad segments:\n";
    for (const auto& segment : adSegments) {
        std::cout << "Ad from " << segment.first << " to " << segment.second << " seconds.\n";
    }

    std::cout << "Video analysis completed.\n";
    return 0;
}
