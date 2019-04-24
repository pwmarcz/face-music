from vision import FaceDetector
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def main():
    fd = FaceDetector()
    fd.start()
    try:
        while True:
            faces = fd.detect()
            if len(faces) > 0:
                logging.info('%d faces: %s', len(faces), faces)
    except KeyboardInterrupt:
        pass
    finally:
        fd.stop()


if __name__ == '__main__':
    main()
